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
    mu_diversity: float = 0.0,
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


class HFWaveCollapseDecoder:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        K: int = 16,
        T: int = 16,
        lambda_interference: float = 0.1,
        gamma_context: float = 1.0,
        tau: float = 0.5,
        repetition_penalty: float = 1.3,
    ) -> None:
        # model: any HuggingFace CausalLM
        # tokenizer: corresponding tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.K = int(K)
        self.T = int(T)
        self.lambda_interference = float(lambda_interference)
        self.gamma_context = float(gamma_context)
        self.tau = float(tau)
        self.repetition_penalty = float(repetition_penalty)

        if self.K <= 0:
            raise ValueError("K must be positive")

        self.device = next(self.model.parameters()).device

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError("model must provide get_input_embeddings() for HFWaveCollapseDecoder")

    @torch.no_grad()
    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Use output_hidden_states=True
        # Return last hidden layer: (1, L, D)
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    @torch.no_grad()
    def _iterative_collapse(
        self,
               logits: torch.Tensor,
               hidden_states: torch.Tensor,
               generated_ids: set | None = None,
    ) -> int:
        # Same algorithm as WaveCollapseDecoder v3
        # K candidates, T iterations
        # Returns token id
        next_logits = logits  # (V,)

        top_logits, top_idx = torch.topk(next_logits, k=min(self.K, next_logits.numel()), dim=-1)

        if generated_ids:
            top_ids_list = top_idx.tolist()
            for i, tok_id in enumerate(top_ids_list):
                if tok_id in generated_ids:
                    top_logits[i] = top_logits[i] / self.repetition_penalty

        emb = self.model.get_input_embeddings()
        candidate_states = emb(top_idx)  # (K, D)
        candidate_states = F.normalize(candidate_states.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

        # Context vector from last few hidden states
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
            mu_diversity=0.0,
        )

        next_id = int(top_idx[A.argmax(dim=-1)].item())
        return next_id

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        # Tokenize prompt
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        generated_ids: set[int] = set()

        # Autoregressive generation with wave collapse
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            logits = outputs.logits  # (1, L, V)
            hidden_states = outputs.hidden_states[-1]  # (1, L, D)

            next_logits = logits[0, -1]  # (V,)
            next_id = self._iterative_collapse(next_logits, hidden_states, generated_ids)

            next_token = torch.tensor([[next_id]], device=self.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_ids.add(next_id)

        # Return decoded string
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

