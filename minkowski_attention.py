"""minkowski_attention.py

MinkowskiAttention: a drop-in replacement for PyTorch's nn.MultiheadAttention that
computes attention weights using a Minkowski spacetime interval rather than an
Euclidean similarity (dot product).

Physics intuition
-----------------
In special relativity, the Minkowski interval between two events i and j is:

    s^2 = -α (Δt)^2 + β (Δx^2 + Δy^2 + Δz^2)

with signature (-,+,+,+). Events with:
- s^2 > 0  are spacelike separated (outside each other's light cone)
- s^2 <= 0 are timelike or lightlike (inside/on the light cone)

This module treats tokens as events in a learned spacetime:
- Each token gets deterministic 4D coordinates (t, x, y, z) from its position and
  embedding.
- Tokens outside the light cone of the query token receive zero attention.
- Tokens inside the light cone receive attention via softmax over s^2.

Note: This replaces scaled dot-product similarity with a geometric causal mask.
The value aggregation remains standard (weighted sum of projected values).

No external dependencies beyond PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _canonicalize_attn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    batch_first: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Convert inputs to (seq_len, batch, embed) layout.

    Returns (q, k, v, transposed) where transposed indicates whether we
    transposed from batch_first.
    """
    if batch_first:
        # (N, L, E) -> (L, N, E)
        return query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), True
    return query, key, value, False


def _undo_batch_first(
    x: torch.Tensor,
    transposed: bool,
) -> torch.Tensor:
    if transposed:
        return x.transpose(0, 1)
    return x


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(L, N, E) -> (N, H, L, D)."""
    l, n, e = x.shape
    head_dim = e // num_heads
    x = x.contiguous().view(l, n, num_heads, head_dim)
    return x.permute(1, 2, 0, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(N, H, L, D) -> (L, N, E)."""
    n, h, l, d = x.shape
    x = x.permute(2, 0, 1, 3).contiguous().view(l, n, h * d)
    return x


def _chunk_means(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute three scalars per token from an embedding by chunking features.

    x: (..., E)

    Returns three tensors of shape (...,) representing (x, y, z) coordinates.

    If E is not divisible by 3, we distribute remainder to earlier chunks.
    """
    e = x.shape[-1]
    a = (e + 2) // 3  # ceil(e/3)
    b = (e + 1) // 3
    c = e // 3
    # Ensure a + b + c == e
    c = e - (a + b)
    x1, x2, x3 = torch.split(x, [a, b, c], dim=-1)
    # Mean over feature chunk -> scalar
    return x1.mean(dim=-1), x2.mean(dim=-1), x3.mean(dim=-1)


@dataclass
class MinkowskiCoords:
    t: torch.Tensor  # (L,) or (S,)
    x: torch.Tensor  # (N, S)
    y: torch.Tensor  # (N, S)
    z: torch.Tensor  # (N, S)


def _coords_from_embedding(
    emb: torch.Tensor,
    seq_len: int,
) -> MinkowskiCoords:
    """Derive deterministic 4D coordinates (t,x,y,z) from (S, N, E) embedding."""
    # Time coordinate from position index.
    # Use float32 for stable interval computation even if emb is fp16/bf16.
    device = emb.device
    t = torch.arange(seq_len, device=device, dtype=torch.float32) / float(seq_len)

    # Space coordinates from normalized embedding.
    # emb: (S, N, E) -> (N, S, E)
    emb_n = emb.permute(1, 0, 2).to(dtype=torch.float32)
    emb_n = F.normalize(emb_n, p=2, dim=-1, eps=1e-12)
    cx, cy, cz = _chunk_means(emb_n)
    return MinkowskiCoords(t=t, x=cx, y=cy, z=cz)


class MinkowskiAttention(nn.Module):
    """Multi-head attention with Minkowski light-cone masking.

    This module is intended as a drop-in replacement for `torch.nn.MultiheadAttention`.

    Key differences vs standard attention:
    - Similarity logits are replaced by Minkowski intervals derived from token
      coordinates.
    - Tokens that are spacelike separated (s^2 > 0) receive exactly zero weight.

    Forward signature matches `nn.MultiheadAttention.forward` for common usage.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if add_bias_kv or add_zero_attn:
            raise NotImplementedError(
                "MinkowskiAttention currently does not implement add_bias_kv/add_zero_attn."
            )
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Learnable metric scalars
        self.alpha = nn.Parameter(torch.tensor(1.0, **factory_kwargs))
        self.beta = nn.Parameter(torch.tensor(1.0, **factory_kwargs))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute MinkowskiAttention.

        Shapes follow `nn.MultiheadAttention`:
        - If batch_first=False: query/key/value: (L, N, E)
        - If batch_first=True : query/key/value: (N, L, E)

        Returns:
        - attn_output: same layout as query
        - attn_output_weights:
            - if need_weights=False: None
            - else shape is (N, L, S) if average_attn_weights else (N, H, L, S)
        """
        if is_causal:
            raise NotImplementedError(
                "MinkowskiAttention does not currently implement is_causal. "
                "Use an explicit attn_mask instead."
            )

        q, k, v, transposed = _canonicalize_attn_inputs(query, key, value, self.batch_first)
        # q,k,v: (L, N, E) / (S, N, E)
        tgt_len, batch_size, _ = q.shape
        src_len = k.shape[0]

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        qh = _split_heads(q_proj, self.num_heads)  # (N, H, L, D)
        kh = _split_heads(k_proj, self.num_heads)  # (N, H, S, D)
        vh = _split_heads(v_proj, self.num_heads)  # (N, H, S, D)

        # Coordinates derived from position and (key) embedding, deterministically.
        q_coords = _coords_from_embedding(q, tgt_len)
        k_coords = _coords_from_embedding(k, src_len)

        # Build Minkowski interval logits: (N, L, S)
        # dt depends only on positions (shared across batch)
        dt = q_coords.t.view(tgt_len, 1) - k_coords.t.view(1, src_len)  # (L, S)
        dt2 = dt * dt

        # dx/dy/dz depend on embeddings (per batch)
        # q_coords.x: (N, L), k_coords.x: (N, S)
        dx = q_coords.x.unsqueeze(-1) - k_coords.x.unsqueeze(-2)  # (N, L, S)
        dy = q_coords.y.unsqueeze(-1) - k_coords.y.unsqueeze(-2)
        dz = q_coords.z.unsqueeze(-1) - k_coords.z.unsqueeze(-2)

        # Broadcast dt2 to batch
        dt2_b = dt2.unsqueeze(0)  # (1, L, S)

        alpha = self.alpha.to(dtype=torch.float32)
        beta = self.beta.to(dtype=torch.float32)

        s2 = (-alpha * dt2_b) + (beta * (dx * dx + dy * dy + dz * dz))  # (N, L, S)

        # Light cone mask: outside (spacelike) => weight 0.
        # Implement via -inf logits so softmax gives exactly 0.
        logits = s2.masked_fill(s2 > 0.0, float("-inf"))

        # Apply additional masks (attn_mask, key_padding_mask)
        if attn_mask is not None:
            # PyTorch allows (L,S), (N*H,L,S), (N,L,S), and bool/float masks.
            # We support (L,S) and (N,L,S) for simplicity.
            if attn_mask.dtype == torch.bool:
                # True means "masked" in PyTorch convention.
                if attn_mask.dim() == 2:
                    logits = logits.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
                elif attn_mask.dim() == 3:
                    logits = logits.masked_fill(attn_mask, float("-inf"))
                else:
                    raise ValueError("attn_mask must be 2D or 3D when boolean")
            else:
                # Additive mask: add to logits
                if attn_mask.dim() == 2:
                    logits = logits + attn_mask.to(dtype=logits.dtype).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    logits = logits + attn_mask.to(dtype=logits.dtype)
                else:
                    raise ValueError("attn_mask must be 2D or 3D when additive")

        if key_padding_mask is not None:
            # key_padding_mask: (N, S) with True indicating pad that should be masked
            if key_padding_mask.shape != (batch_size, src_len):
                raise ValueError(
                    f"key_padding_mask must have shape (N, S) = ({batch_size}, {src_len})"
                )
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))

        # Expand logits across heads (same geometric weights per head).
        attn_logits = logits.unsqueeze(1).expand(batch_size, self.num_heads, tgt_len, src_len)

        # Softmax in fp32, then cast for matmul.
        attn_weights = F.softmax(attn_logits.to(dtype=torch.float32), dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_weights = attn_weights.to(dtype=vh.dtype)

        # Weighted sum: (N,H,L,S) @ (N,H,S,D) -> (N,H,L,D)
        attn_out = torch.matmul(attn_weights, vh)

        # Merge heads and project
        attn_out = _merge_heads(attn_out)  # (L, N, E)
        attn_out = self.out_proj(attn_out)
        attn_out = _undo_batch_first(attn_out, transposed)

        if not need_weights:
            return attn_out, None

        if average_attn_weights:
            # (N,H,L,S) -> (N,L,S)
            w = attn_weights.to(dtype=torch.float32).mean(dim=1)
        else:
            w = attn_weights.to(dtype=torch.float32)

        return attn_out, w


class StandardMultiheadAttention(nn.Module):
    """Baseline wrapper around `nn.MultiheadAttention` with the same interface.

    This exists to make side-by-side comparisons (sparsity, timing) easy.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
