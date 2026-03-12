import time
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: (K,)
    p = F.softmax(logits.to(dtype=torch.float32), dim=-1).clamp_min(1e-12)
    return -(p * p.log()).sum()


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
    def __init__(self, model: nn.Module, K: int = 5, lambda_interference: float = 0.3) -> None:
        self.model = model
        self.K = int(K)
        self.lambda_interference = float(lambda_interference)

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

        for _ in range(max_new_tokens):
            logits = self.model(tokens)  # (1, L, V)
            next_logits = logits[0, -1]  # (V,)

            probs = F.softmax(next_logits.to(dtype=torch.float32), dim=-1)
            top_p, top_idx = torch.topk(probs, k=min(self.K, probs.numel()), dim=-1)

            # Amplitudes A(k)
            A = top_p  # (K,)

            # Embeddings for interference
            emb = self.model.tok_emb(top_idx)  # (K, D)
            emb = F.normalize(emb.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

            # Pairwise cosine similarity matrix: (K, K)
            I = emb @ emb.t()
            I.fill_diagonal_(0.0)

            # Wave weight W(k) = A(k) + lambda * sum_j I(k,j)*A(j)
            W = A + self.lambda_interference * (I @ A)

            next_id = int(top_idx[W.argmax(dim=-1)].item())
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device, dtype=tokens.dtype)], dim=1)

        return tokens[0].tolist()


@torch.no_grad()
def wave_collapse_step_stats(model: nn.Module, tokens: torch.Tensor, K: int, lambda_interference: float):
    """Compute next-token selection stats for WaveCollapse decoding.

    Returns:
        next_id: int
        entropy_W: float  (entropy of softmax(W))
        top_ids: (K,) LongTensor
    """
    logits = model(tokens)  # (1, L, V)
    next_logits = logits[0, -1]  # (V,)

    probs = F.softmax(next_logits.to(dtype=torch.float32), dim=-1)
    top_p, top_idx = torch.topk(probs, k=min(K, probs.numel()), dim=-1)

    A = top_p  # (K,)
    emb = model.tok_emb(top_idx)  # (K, D)
    emb = F.normalize(emb.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)

    I = emb @ emb.t()
    I.fill_diagonal_(0.0)

    W = A + lambda_interference * (I @ A)

    entropy_W = float(_entropy_from_logits(W).item())
    next_id = int(top_idx[W.argmax(dim=-1)].item())

    return next_id, entropy_W, top_idx
