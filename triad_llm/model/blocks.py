from typing import Optional

import torch
from torch import nn

from triad_llm.attention import MinkowskiAttention


class MinkowskiTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MinkowskiAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        h = self.ln1(x)
        # Prefer built-in causal path when available; fall back to an explicit triangular mask
        # for older PyTorch versions.
        if is_causal:
            try:
                attn_out, _ = self.attn(
                    h,
                    h,
                    h,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    is_causal=True,
                )
            except TypeError:
                l = h.shape[1]
                causal_mask = torch.triu(
                    torch.ones((l, l), device=h.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_out, _ = self.attn(
                    h,
                    h,
                    h,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask=causal_mask,
                )
        else:
            attn_out, _ = self.attn(
                h,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x
