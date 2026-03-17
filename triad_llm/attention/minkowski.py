"""MinkowskiAttention and a standard MultiheadAttention baseline.

MinkowskiAttention replaces scaled dot-product similarity with a Minkowski
spacetime interval. Tokens are treated as events with (t,x,y,z) coordinates.
Attention is only allowed inside the light cone (timelike/lightlike separation).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .coords import coords_from_embedding


def _canonicalize_attn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    batch_first: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if batch_first:
        return query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), True
    return query, key, value, False


def _undo_batch_first(x: torch.Tensor, transposed: bool) -> torch.Tensor:
    if transposed:
        return x.transpose(0, 1)
    return x


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    l, n, e = x.shape
    head_dim = e // num_heads
    x = x.contiguous().view(l, n, num_heads, head_dim)
    return x.permute(1, 2, 0, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    n, h, l, d = x.shape
    x = x.permute(2, 0, 1, 3).contiguous().view(l, n, h * d)
    return x


class MinkowskiAttention(nn.Module):
    """Multi-head attention with Minkowski light-cone masking.

    Signature is intended to match `torch.nn.MultiheadAttention.forward` for
    common usage.

    Attention weights are computed from the Minkowski interval:

        s^2 = -α (Δt)^2 + β (Δx^2 + Δy^2 + Δz^2)

    Spacelike separations (s^2 > 0) are outside the light cone and are masked
    to zero weight.
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

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.spatial_proj = nn.Linear(embed_dim, 3, bias=True, **factory_kwargs)
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spatial_proj.bias)

        self.time_scale = nn.Parameter(torch.tensor(2.0, **factory_kwargs))

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
        q, k, v, transposed = _canonicalize_attn_inputs(query, key, value, self.batch_first)
        tgt_len, batch_size, _ = q.shape
        src_len = k.shape[0]

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        vh = _split_heads(v_proj, self.num_heads)  # (N, H, S, D)

        q_for_coords = q if q.shape[-1] == self.embed_dim else q_proj
        k_for_coords = k if k.shape[-1] == self.embed_dim else k_proj

        q_coords = coords_from_embedding(
            q_for_coords, tgt_len, spatial_proj=self.spatial_proj, time_scale=self.time_scale
        )
        k_coords = coords_from_embedding(
            k_for_coords, src_len, spatial_proj=self.spatial_proj, time_scale=self.time_scale
        )

        dt = q_coords.t.view(tgt_len, 1) - k_coords.t.view(1, src_len)  # (L, S)
        dt2 = dt * dt

        dx = q_coords.x.unsqueeze(-1) - k_coords.x.unsqueeze(-2)  # (N, L, S)
        dy = q_coords.y.unsqueeze(-1) - k_coords.y.unsqueeze(-2)
        dz = q_coords.z.unsqueeze(-1) - k_coords.z.unsqueeze(-2)

        dt2_b = dt2.unsqueeze(0)  # (1, L, S)

        alpha = self.alpha.to(dtype=torch.float32)
        beta = self.beta.to(dtype=torch.float32)

        s2 = (-alpha * dt2_b) + (beta * (dx * dx + dy * dy + dz * dz))  # (N, L, S)
        logits = s2.masked_fill(s2 > 0.0, float("-inf"))

        if is_causal:
            # Combine physical Minkowski masking with logical causal masking (j > i).
            # Mask is created directly on the logits device.
            causal_mask = torch.triu(
                torch.ones((tgt_len, src_len), device=logits.device, dtype=torch.bool),
                diagonal=1,
            )  # (L, S)
            logits.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    logits.masked_fill_(attn_mask.unsqueeze(0), float("-inf"))
                elif attn_mask.dim() == 3:
                    logits.masked_fill_(attn_mask, float("-inf"))
                else:
                    raise ValueError("attn_mask must be 2D or 3D when boolean")
            else:
                if attn_mask.dim() == 2:
                    logits = logits + attn_mask.to(dtype=logits.dtype).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    logits = logits + attn_mask.to(dtype=logits.dtype)
                else:
                    raise ValueError("attn_mask must be 2D or 3D when additive")

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, src_len):
                raise ValueError(
                    f"key_padding_mask must have shape (N, S) = ({batch_size}, {src_len})"
                )
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))

        attn_logits = logits.unsqueeze(1).expand(batch_size, self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_logits.to(dtype=torch.float32), dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_weights = attn_weights.to(dtype=vh.dtype)

        attn_out = torch.matmul(attn_weights, vh)  # (N,H,L,D)
        attn_out = _merge_heads(attn_out)  # (L, N, E)
        attn_out = self.out_proj(attn_out)
        attn_out = _undo_batch_first(attn_out, transposed)

        if not need_weights:
            return attn_out, None

        if average_attn_weights:
            w = attn_weights.to(dtype=torch.float32).mean(dim=1)
        else:
            w = attn_weights.to(dtype=torch.float32)

        return attn_out, w


class StandardMultiheadAttention(nn.Module):
    """Baseline wrapper around `nn.MultiheadAttention` with the same interface."""

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
