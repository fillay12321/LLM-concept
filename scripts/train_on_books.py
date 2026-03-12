import argparse
import os
import time
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from triad_llm.decoding import GreedyDecoder, WaveCollapseDecoder
from triad_llm.model import MinkowskiTransformer, StandardTransformer
from triad_llm.training.book_dataset import BookDataset
from triad_llm.training.tokenizer import TiktokenWrapper


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    b, l, v = logits.shape
    return F.cross_entropy(logits.reshape(b * l, v), targets.reshape(b * l))


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _ascii_loss_curve(losses: List[float], *, title: str) -> None:
    if len(losses) == 0:
        return

    width = 60
    height = 10

    if len(losses) <= width:
        samples = losses
    else:
        idx = torch.linspace(0, len(losses) - 1, steps=width).round().to(torch.long)
        samples = [losses[int(i.item())] for i in idx]

    lo = min(samples)
    hi = max(samples)
    rng = max(hi - lo, 1e-12)

    levels = []
    for v in samples:
        # Map to [0, height-1]
        levels.append(int(round((v - lo) / rng * (height - 1))))

    print(f"Loss curve ({title}):")
    for row in range(height - 1, -1, -1):
        y = lo + (rng * row / float(height - 1))
        line = []
        for lv in levels:
            if lv > row:
                ch = "█"
            elif lv == row:
                ch = "▄"
            elif lv == row - 1:
                ch = "▂"
            else:
                ch = " "
            line.append(ch)
        print(f"{y:>5.2f} |{''.join(line)}")


def _format_wall(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@torch.no_grad()
def _eval_perplexity(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.to(torch.long)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = model(inp)
        loss = _lm_loss(logits, tgt)
        losses.append(float(loss.item()))

    mean_loss = sum(losses) / max(len(losses), 1)
    return float(torch.exp(torch.tensor(mean_loss)).item())


@torch.no_grad()
def _mean_sequence_coherence(model: nn.Module, loader: DataLoader, tok: TiktokenWrapper, *, n_prompts: int = 32) -> Tuple[float, float]:
    # Compare Greedy vs Wave coherence on prompts drawn from val set.
    greedy = GreedyDecoder(model)
    wave = WaveCollapseDecoder(
        model,
        K=10,
        T=3,
        lambda_interference=0.1,
        tau=0.5,
        gamma_context=1.0,
        mu_diversity=0.0,
    )

    def pairwise_cos_sim(ids: List[int]) -> float:
        x = model.tok_emb(torch.tensor(ids, dtype=torch.long))
        x = F.normalize(x.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)
        sim = x @ x.t()
        n = sim.shape[0]
        if n <= 1:
            return 1.0
        mask = ~torch.eye(n, dtype=torch.bool)
        return float(sim[mask].mean().item())

    g_scores = []
    w_scores = []

    seen = 0
    for batch in loader:
        if seen >= n_prompts:
            break
        batch = batch.to(torch.long)
        for i in range(batch.shape[0]):
            if seen >= n_prompts:
                break
            prompt = batch[i : i + 1, :20].contiguous()
            g_ids = greedy.generate(prompt, max_new_tokens=50)
            w_ids = wave.generate(prompt, max_new_tokens=50)
            g_scores.append(pairwise_cos_sim(g_ids))
            w_scores.append(pairwise_cos_sim(w_ids))
            seen += 1

    g = float(sum(g_scores) / max(len(g_scores), 1))
    w = float(sum(w_scores) / max(len(w_scores), 1))
    return g, w


def _generation_sample(
    *,
    model: nn.Module,
    tokenizer: TiktokenWrapper,
    batch: torch.Tensor,
    batch_idx: int,
) -> None:
    prompt_ids = batch[0, :20].tolist()
    prompt_txt = tokenizer.decode(prompt_ids)

    greedy = GreedyDecoder(model)
    wave = WaveCollapseDecoder(
        model,
        K=10,
        T=3,
        gamma_context=1.0,
        lambda_interference=0.1,
        tau=0.5,
        mu_diversity=0.0,
    )

    seed = torch.tensor([prompt_ids], dtype=torch.long)
    g_ids = greedy.generate(seed, max_new_tokens=50)
    w_ids = wave.generate(seed, max_new_tokens=50)

    g_new = g_ids[len(prompt_ids) :]
    w_new = w_ids[len(prompt_ids) :]

    print("-" * 70)
    print(f"=== Generation sample @ batch {batch_idx} ===")
    print(f"Prompt: {prompt_txt}")
    print(f"Greedy:  {tokenizer.decode(g_new)}")
    print(f"Wave:    {tokenizer.decode(w_new)}")
    print("=" * 70)


def _train_epoch(
    *,
    name: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    train_loader: DataLoader,
    epoch: int,
    epochs: int,
    tokenizer: TiktokenWrapper,
    global_batch_offset: int,
) -> Tuple[float, float, List[float], int]:
    model.train()
    t0 = time.perf_counter()
    losses: List[float] = []

    num_batches = len(train_loader)
    epoch_start = time.perf_counter()

    for step, batch in enumerate(train_loader, start=1):
        batch = batch.to(torch.long)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        logits = model(inp)
        loss = _lm_loss(logits, tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))

        now = time.perf_counter()
        elapsed = now - epoch_start
        bat_s = step / max(elapsed, 1e-12)
        eta_s = (num_batches - step) / max(bat_s, 1e-12)
        msg = (
            f"Epoch {epoch}/{epochs} | Batch {step}/{num_batches} | loss={loss.item():.4f} | "
            f"{bat_s:.1f} bat/s | ETA {_format_eta(eta_s)}"
        )
        print("\r" + msg, end="", flush=True)

        global_batch = global_batch_offset + step
        if global_batch % 500 == 0:
            print()
            model.eval()
            with torch.no_grad():
                _generation_sample(model=model, tokenizer=tokenizer, batch=batch, batch_idx=global_batch)
            model.train()
            epoch_start = time.perf_counter()

    print()

    mean_loss = float(sum(losses) / max(len(losses), 1))
    t1 = time.perf_counter()
    return mean_loss, (t1 - t0), losses, global_batch_offset + num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Text files to train on")
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cpu")

    seq_len = 256
    stride = 128

    tok = TiktokenWrapper()
    ds = BookDataset(file_paths=args.files, tokenizer=tok, seq_len=seq_len, stride=stride)
    train_ds, val_ds = ds.train_val_split(val_ratio=0.1)

    print("Dataset stats:")
    print(f"  Total tokens: {ds.all_tokens.numel():,}")
    print(f"  Total sequences: {len(ds):,}")
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences: {len(val_ds):,}")
    print(f"  Vocab size: {tok.vocab_size:,}")

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    cfg = {
        "vocab_size": tok.vocab_size,
        "max_seq_len": seq_len,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.1,
    }

    mink = MinkowskiTransformer(**cfg).to(device)
    std = StandardTransformer(**cfg).to(device)

    print(f"Minkowski params: {sum(p.numel() for p in mink.parameters()):,}")
    print(f"Standard params: {sum(p.numel() for p in std.parameters()):,}")

    lr = 3e-4
    epochs = 3

    os.makedirs("checkpoints", exist_ok=True)
    mink_opt = torch.optim.AdamW(mink.parameters(), lr=lr)
    std_opt = torch.optim.AdamW(std.parameters(), lr=lr)

    mink_best_loss = float("inf")
    std_best_loss = float("inf")
    mink_best_ppl = float("inf")
    std_best_ppl = float("inf")
    mink_best_epoch = 0
    std_best_epoch = 0

    mink_loss_history: List[float] = []
    std_loss_history: List[float] = []

    mink_global_batches = 0
    std_global_batches = 0

    for epoch in range(1, epochs + 1):
        print(f"\nTraining epoch {epoch}/{epochs}...")

        mink_mean_loss, mink_time_s, mink_epoch_losses, mink_global_batches = _train_epoch(
            name="Minkowski",
            model=mink,
            opt=mink_opt,
            train_loader=train_loader,
            epoch=epoch,
            epochs=epochs,
            tokenizer=tok,
            global_batch_offset=mink_global_batches,
        )
        mink_loss_history.extend(mink_epoch_losses)
        mink_best_loss = min(mink_best_loss, mink_mean_loss)

        std_mean_loss, std_time_s, std_epoch_losses, std_global_batches = _train_epoch(
            name="Standard",
            model=std,
            opt=std_opt,
            train_loader=train_loader,
            epoch=epoch,
            epochs=epochs,
            tokenizer=tok,
            global_batch_offset=std_global_batches,
        )
        std_loss_history.extend(std_epoch_losses)
        std_best_loss = min(std_best_loss, std_mean_loss)

        mink_ppl = _eval_perplexity(mink, val_loader)
        std_ppl = _eval_perplexity(std, val_loader)

        if mink_ppl < mink_best_ppl:
            mink_best_ppl = mink_ppl
            mink_best_epoch = epoch
        if std_ppl < std_best_ppl:
            std_best_ppl = std_ppl
            std_best_epoch = epoch

        mink_ckpt_path = os.path.join("checkpoints", f"mink_epoch{epoch}.pt")
        torch.save(
            {
                "model_type": "mink",
                "config": cfg,
                "state_dict": mink.state_dict(),
                "encoding_name": tok.encoding_name,
            },
            mink_ckpt_path,
        )

        std_ckpt_path = os.path.join("checkpoints", f"std_epoch{epoch}.pt")
        torch.save(
            {
                "model_type": "std",
                "config": cfg,
                "state_dict": std.state_dict(),
                "encoding_name": tok.encoding_name,
            },
            std_ckpt_path,
        )

        _ascii_loss_curve(mink_loss_history, title="Minkowski")
        _ascii_loss_curve(std_loss_history, title="Standard")

        bar = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print(bar)
        print(f"Epoch {epoch}/{epochs} complete")
        print(
            f"Minkowski: mean_loss={mink_mean_loss:.4f} | time={_format_wall(mink_time_s)} | best_loss={mink_best_loss:.4f}"
        )
        print(
            f"Standard:  mean_loss={std_mean_loss:.4f} | time={_format_wall(std_time_s)} | best_loss={std_best_loss:.4f}"
        )
        print(f"Val perplexity: Minkowski={mink_ppl:.3f} | Standard={std_ppl:.3f}")
        print(bar)
        print(f"Saved checkpoint: {mink_ckpt_path}")
        print(f"Saved checkpoint: {std_ckpt_path}")

    print("\nTRAINING COMPLETE")
    print(f"Best Minkowski perplexity: {mink_best_ppl:.2f} (epoch {mink_best_epoch})")
    print(f"Best Standard perplexity: {std_best_ppl:.2f} (epoch {std_best_epoch})")

    g_coh, w_coh = _mean_sequence_coherence(mink, val_loader, tok)
    print(f"Wave vs Greedy coherence on val set: {w_coh:.3f} vs {g_coh:.3f}")

    winner = "Minkowski" if mink_best_ppl < std_best_ppl else "Standard"
    print(f"Winner: {winner} (by perplexity)")


if __name__ == "__main__":
    main()
