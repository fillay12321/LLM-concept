import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrainStats:
    mean_loss: float
    wall_time_s: float


def _iterate_minibatches(x: torch.Tensor, batch_size: int, *, shuffle: bool):
    n = x.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield x[idx[i : i + batch_size]]


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    b, l, v = logits.shape
    return F.cross_entropy(logits.reshape(b * l, v), targets.reshape(b * l))


def train_language_model(
    model: nn.Module,
    train_seqs: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    epoch_losses = []
    epoch_stats = []

    num_batches = (train_seqs.shape[0] + batch_size - 1) // batch_size

    for epoch_idx in range(epochs):
        print(f"Epoch {epoch_idx + 1}/{epochs} starting...")
        t0 = time.perf_counter()
        losses = []

        for batch_idx, batch in enumerate(_iterate_minibatches(train_seqs, batch_size, shuffle=True), start=1):
            inp = batch[:, :-1]
            tgt = batch[:, 1:]

            logits = model(inp)
            loss = _lm_loss(logits, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if batch_idx % 50 == 0 or batch_idx == num_batches:
                print(f"  Batch {batch_idx}/{num_batches} | loss={loss.item():.4f}")

        t1 = time.perf_counter()
        mean_loss = float(sum(losses) / max(len(losses), 1))
        epoch_losses.append(mean_loss)
        epoch_stats.append(TrainStats(mean_loss=mean_loss, wall_time_s=t1 - t0))

        print(f"Epoch {epoch_idx + 1}/{epochs} done | mean_loss={mean_loss:.4f} | time={t1 - t0:.1f}s")

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
