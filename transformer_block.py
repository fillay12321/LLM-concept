import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from minkowski_attention import MinkowskiAttention


def _causal_attn_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # Boolean mask with True indicating positions that should be masked.
    # Shape: (L, L)
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


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
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        # Pre-LN FFN
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
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x


class MinkowskiTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [MinkowskiTransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L)
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device = tokens.device
        pos = torch.arange(seq_len, device=device)

        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        attn_mask = _causal_attn_mask(seq_len, device=device)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class StandardTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [StandardTransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device = tokens.device
        pos = torch.arange(seq_len, device=device)

        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        attn_mask = _causal_attn_mask(seq_len, device=device)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


@dataclass
class TrainStats:
    mean_loss: float
    wall_time_s: float


def _iterate_minibatches(x: torch.Tensor, batch_size: int, *, shuffle: bool) -> torch.Tensor:
    n = x.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield x[idx[i : i + batch_size]]


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (B, L, V), targets: (B, L)
    b, l, v = logits.shape
    return F.cross_entropy(logits.reshape(b * l, v), targets.reshape(b * l))


def train_language_model(
    model: nn.Module,
    train_seqs: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Tuple[list[float], list[TrainStats]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    epoch_losses: list[float] = []
    epoch_stats: list[TrainStats] = []

    for _ in range(epochs):
        t0 = time.perf_counter()
        losses = []
        for batch in _iterate_minibatches(train_seqs, batch_size, shuffle=True):
            # Next-token prediction
            inp = batch[:, :-1]
            tgt = batch[:, 1:]

            logits = model(inp)
            loss = _lm_loss(logits, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        t1 = time.perf_counter()
        mean_loss = float(sum(losses) / max(len(losses), 1))
        epoch_losses.append(mean_loss)
        epoch_stats.append(TrainStats(mean_loss=mean_loss, wall_time_s=t1 - t0))

    return epoch_losses, epoch_stats


@torch.no_grad()
def eval_perplexity(model: nn.Module, eval_seqs: torch.Tensor, batch_size: int) -> float:
    model.eval()
    losses = []
    for batch in _iterate_minibatches(eval_seqs, batch_size, shuffle=False):
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = model(inp)
        loss = _lm_loss(logits, tgt)
        losses.append(loss.item())

    mean_loss = float(sum(losses) / max(len(losses), 1))
    return float(math.exp(mean_loss))


def main():
    torch.manual_seed(42)

    # Config
    d_model = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    vocab_size = 50
    seq_len = 64

    train_n = 10_000
    val_n = 1_000
    batch_size = 32
    epochs = 5
    lr = 1e-3

    device = torch.device("cpu")

    # Synthetic dataset: random integer tokens.
    data = torch.randint(0, vocab_size, (train_n + val_n, seq_len), device=device)
    train_seqs = data[:train_n]
    val_seqs = data[train_n:]

    mink = MinkowskiTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    std = StandardTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    mink_losses, _ = train_language_model(mink, train_seqs, epochs, batch_size, lr)
    std_losses, _ = train_language_model(std, train_seqs, epochs, batch_size, lr)

    for i in range(epochs):
        print(f"Epoch {i + 1}: Minkowski loss={mink_losses[i]:.4f} | Standard loss={std_losses[i]:.4f}")

    mink_ppl = eval_perplexity(mink, val_seqs, batch_size=batch_size)
    std_ppl = eval_perplexity(std, val_seqs, batch_size=batch_size)

    print(f"Held-out perplexity: Minkowski={mink_ppl:.3f} | Standard={std_ppl:.3f}")


if __name__ == "__main__":
    main()
