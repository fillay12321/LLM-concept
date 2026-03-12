from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    p = p.to(dtype=torch.float32).clamp_min(1e-12)
    return -(p * p.log()).sum()


def _iterative_collapse(
    *,
    base_logits: torch.Tensor,
    candidate_states: torch.Tensor,
    context_vec: torch.Tensor,
    T: int,
    lambda_interference: float,
    tau: float = 0.5,
    gamma_context: float,
    mu_diversity: float,
    convergence_eps: float = 1e-4,
) -> Tuple[torch.Tensor, int]:
    # base_logits: (K,)
    # candidate_states: (K, D) normalized
    # context_vec: (D,) normalized

    K = base_logits.shape[0]

    # C(k) = cos(candidate, context)
    C = torch.matmul(candidate_states, context_vec)  # (K,)

    # Full interference: similar=constructive, dissimilar=destructive.
    # Temperature scaling controls interference sharpness.
    I = candidate_states @ candidate_states.t()  # (K, K), range [-1, 1]
    I = I / float(tau)
    I.fill_diagonal_(0.0)

    # Initial amplitudes from logits restricted to top-K
    A = F.softmax(base_logits.to(dtype=torch.float32), dim=-1)  # (K,)

    iters_to_conv = 0
    for t in range(int(T)):
        prev = A

        # Normalized interference prevents runaway amplification from positive feedback.
        raw_interference = I @ A  # (K,)
        norm = I.abs().sum(dim=1).clamp_min(1e-12)  # (K,)
        interference = raw_interference / norm

        A = A + float(lambda_interference) * interference
        A = A + float(gamma_context) * C

        A = F.softmax(A, dim=-1)

        max_change = (A - prev).abs().max().item()
        iters_to_conv = t + 1
        if max_change < convergence_eps:
            break

    return A, iters_to_conv


class GreedyDecoder:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def generate(self, seed_tokens: torch.Tensor, max_new_tokens: int = 32) -> List[int]:
        # seed_tokens: (1, L)
        if seed_tokens.dim() != 2 or seed_tokens.shape[0] != 1:
            raise ValueError("seed_tokens must have shape (1, L)")

        device = next(self.model.parameters()).device
        tokens = seed_tokens.to(device)

        for _ in range(max_new_tokens):
            logits = self.model(tokens)  # (1, L, V)
            next_logits = logits[0, -1]  # (V,)
            next_id = int(next_logits.argmax(dim=-1).item())
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device, dtype=tokens.dtype)], dim=1)

        return tokens[0].tolist()


class WaveCollapseDecoder:
    def __init__(
        self,
        model: nn.Module,
        K: int = 10,
        T: int = 5,
        lambda_interference: float = 0.3,
        tau: float = 0.5,
        gamma_context: float = 0.2,
        mu_diversity: float = 0.1,
    ) -> None:
        self.model = model
        self.K = int(K)
        self.T = int(T)
        self.lambda_interference = float(lambda_interference)
        self.tau = float(tau)
        self.gamma_context = float(gamma_context)
        self.mu_diversity = float(mu_diversity)

        if self.K <= 0:
            raise ValueError("K must be positive")

    @torch.no_grad()
    def generate(self, seed_tokens: torch.Tensor, max_new_tokens: int = 32) -> List[int]:
        # seed_tokens: (1, L)
        if seed_tokens.dim() != 2 or seed_tokens.shape[0] != 1:
            raise ValueError("seed_tokens must have shape (1, L)")

        device = next(self.model.parameters()).device
        tokens = seed_tokens.to(device)

        if not hasattr(self.model, "tok_emb"):
            raise AttributeError("model must have tok_emb for WaveCollapseDecoder")

        if not hasattr(self.model, "get_hidden_states"):
            raise AttributeError("model must implement get_hidden_states(tokens) for WaveCollapseDecoder")

        for _ in range(max_new_tokens):
            logits = self.model(tokens)  # (1, L, V)
            next_logits = logits[0, -1]  # (V,)

            hidden_states = self.model.get_hidden_states(tokens)  # (1, L, D)

            # Top-K candidates by logit value
            top_logits, top_idx = torch.topk(next_logits, k=min(self.K, next_logits.numel()), dim=-1)

            # Candidate states from token embedding table
            candidate_states = self.model.tok_emb(top_idx)  # (K, D)
            candidate_states = F.normalize(candidate_states.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

            # Context vector from last 3 hidden states
            ctx = hidden_states[0, max(hidden_states.shape[1] - 3, 0) :, :].mean(dim=0)
            context_vec = F.normalize(ctx.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

            A, _ = _iterative_collapse(
                base_logits=top_logits,
                candidate_states=candidate_states,
                context_vec=context_vec,
                T=self.T,
                lambda_interference=self.lambda_interference,
                tau=self.tau,
                gamma_context=self.gamma_context,
                mu_diversity=self.mu_diversity,
            )

            next_id = int(top_idx[A.argmax(dim=-1)].item())
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device, dtype=tokens.dtype)], dim=1)

        return tokens[0].tolist()


@torch.no_grad()
def wave_collapse_step_stats(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    K: int,
    T: int,
    lambda_interference: float,
    tau: float = 0.5,
    gamma_context: float,
    mu_diversity: float,
    convergence_eps: float = 1e-4,
):
    """Compute next-token selection stats for WaveCollapse decoding.

    Returns:
        next_id: int
        final_A: (K,) float tensor (softmax-normalized after iterative collapse)
        iters_to_conv: int
        top_ids: (K,) LongTensor
    """
    logits = model(tokens)  # (1, L, V)
    next_logits = logits[0, -1]  # (V,)

    if not hasattr(model, "get_hidden_states"):
        raise AttributeError("model must implement get_hidden_states(tokens) for WaveCollapse decoding")
    hidden_states = model.get_hidden_states(tokens)  # (1, L, D)

    top_logits, top_idx = torch.topk(next_logits, k=min(int(K), next_logits.numel()), dim=-1)

    candidate_states = model.tok_emb(top_idx)
    candidate_states = F.normalize(candidate_states.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

    ctx = hidden_states[0, max(hidden_states.shape[1] - 3, 0) :, :].mean(dim=0)
    context_vec = F.normalize(ctx.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

    final_A, iters_to_conv = _iterative_collapse(
        base_logits=top_logits,
        candidate_states=candidate_states,
        context_vec=context_vec,
        T=int(T),
        lambda_interference=float(lambda_interference),
        tau=float(tau),
        gamma_context=float(gamma_context),
        mu_diversity=float(mu_diversity),
        convergence_eps=float(convergence_eps),
    )

    next_id = int(top_idx[final_A.argmax(dim=-1)].item())
    return next_id, final_A, iters_to_conv, top_idx
