from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MinkowskiCoords:
    t: torch.Tensor  # (L,) or (S,)
    x: torch.Tensor  # (N, S)
    y: torch.Tensor  # (N, S)
    z: torch.Tensor  # (N, S)


def coords_from_embedding(
    emb: torch.Tensor,
    seq_len: int,
    spatial_proj: nn.Linear,
    time_scale: torch.Tensor,
) -> MinkowskiCoords:
    """Derive 4D coordinates (t,x,y,z) from (S, N, E) embedding.

    - t is derived from position index and scaled by a learnable scalar.
    - (x,y,z) are derived from a learned linear projection of the embedding,
      then bounded with tanh to [-1, 1].

    The mapping is deterministic given the module parameters and input embedding.
    """
    device = emb.device
    t0 = torch.arange(seq_len, device=device, dtype=torch.float32) / float(seq_len)
    t = t0 * time_scale.to(dtype=torch.float32)

    emb_f = emb.permute(1, 0, 2).to(dtype=torch.float32)  # (N, S, E)
    xyz = spatial_proj(emb_f)  # (N, S, 3)
    xyz = torch.tanh(xyz)

    return MinkowskiCoords(t=t, x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2])
