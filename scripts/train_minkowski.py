import argparse
import math
import os
import time
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from triad_llm.decoding import GreedyDecoder, WaveCollapseDecoder
from triad_llm.model import MinkowskiTransformer
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
        levels.append(int(round((v - lo) / rng * (height - 1))))

    print(f"Loss curve ({title}):", flush=True)
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
        print(f"{y:>5.2f} |{''.join(line)}", flush=True)


def _format_wall(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@torch.no_grad()
def _eval_perplexity(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.to(device=device, dtype=torch.long)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = model(inp)
        loss = _lm_loss(logits, tgt)
        losses.append(float(loss.item()))

    mean_loss = sum(losses) / max(len(losses), 1)
    return float(torch.exp(torch.tensor(mean_loss)).item())


@torch.no_grad()
def _mean_sequence_coherence(
    model: nn.Module,
    loader: DataLoader,
    tok: TiktokenWrapper,
    device: torch.device,
    *,
    n_prompts: int = 32,
) -> Tuple[float, float]:
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
        x = model.tok_emb(torch.tensor(ids, dtype=torch.long, device=device))
        x = F.normalize(x.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)
        sim = x @ x.t()
        n = sim.shape[0]
        if n <= 1:
            return 1.0
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        return float(sim[mask].mean().item())

    g_scores: List[float] = []
    w_scores: List[float] = []

    seen = 0
    for batch in loader:
        if seen >= n_prompts:
            break
        batch = batch.to(device=device, dtype=torch.long)
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


def _lr_for_step(
    *,
    global_step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr

    step = max(global_step, 1)
    if step <= warmup_steps:
        return base_lr * float(step) / float(max(warmup_steps, 1))

    if step >= total_steps:
        return min_lr

    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (base_lr - min_lr) * cosine)


def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def _generation_sample(
    *,
    model: nn.Module,
    tokenizer: TiktokenWrapper,
    batch: torch.Tensor,
    batch_idx: int,
    device: torch.device,
) -> None:
    prompt_ids = batch[0, :20].tolist()
    prompt_txt = tokenizer.decode(prompt_ids)

    greedy = GreedyDecoder(model)
    wave = WaveCollapseDecoder(
        model,
        K=16,
        T=16,
        lambda_interference=0.1,
        gamma_context=1.0,
        tau=0.5,
        mu_diversity=0.0,
    )

    seed = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    g_ids = greedy.generate(seed, max_new_tokens=80)
    w_ids = wave.generate(seed, max_new_tokens=80)

    g_new = g_ids[len(prompt_ids) :]
    w_new = w_ids[len(prompt_ids) :]

    print("-" * 70, flush=True)
    print(f"=== Generation sample @ batch {batch_idx} ===", flush=True)
    print(f"Prompt: {prompt_txt}", flush=True)
    print(f"Greedy:  {tokenizer.decode(g_new)}", flush=True)
    print(f"Wave:    {tokenizer.decode(w_new)}", flush=True)
    print("=" * 70, flush=True)


def _train_epoch(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    train_loader: DataLoader,
    epoch: int,
    epochs: int,
    tokenizer: TiktokenWrapper,
    global_batch_offset: int,
    global_step_offset: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    device: torch.device,
) -> Tuple[float, float, List[float], int, int]:
    model.train()
    t0 = time.perf_counter()
    losses: List[float] = []

    num_batches = len(train_loader)
    epoch_start = time.perf_counter()

    global_batch = global_batch_offset
    global_step = global_step_offset

    for step, batch in enumerate(train_loader, start=1):
        batch = batch.to(device=device, dtype=torch.long)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        global_step += 1
        lr = _lr_for_step(
            global_step=global_step,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )
        _set_optimizer_lr(opt, lr)

        logits = model(inp)
        loss = _lm_loss(logits, tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(flush=True)
            model.eval()
            with torch.no_grad():
                _generation_sample(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    batch_idx=global_batch,
                    device=device,
                )
            model.train()
            epoch_start = time.perf_counter()

    print(flush=True)

    mean_loss = float(sum(losses) / max(len(losses), 1))
    t1 = time.perf_counter()
    return mean_loss, (t1 - t0), losses, global_batch, global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Text files to train on")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    seq_len = 512
    stride = 256

    tok = TiktokenWrapper()
    ds = BookDataset(file_paths=args.files, tokenizer=tok, seq_len=seq_len, stride=stride)
    train_ds, val_ds = ds.train_val_split(val_ratio=0.1)

    print("Dataset stats:", flush=True)
    print(f"  Total tokens: {ds.all_tokens.numel():,}", flush=True)
    print(f"  Total sequences: {len(ds):,}", flush=True)
    print(f"  Train sequences: {len(train_ds):,}", flush=True)
    print(f"  Val sequences: {len(val_ds):,}", flush=True)
    print(f"  Vocab size: {tok.vocab_size:,}", flush=True)

    batch_size = 8
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    cfg = {
        "vocab_size": tok.vocab_size,
        "max_seq_len": seq_len,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1,
    }

    model = MinkowskiTransformer(**cfg).to(device)
    print(f"Minkowski params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    base_lr = 3e-4
    min_lr = 1e-5
    epochs = 10
    warmup_steps = 500

    os.makedirs("checkpoints", exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

    best_loss = float("inf")
    best_ppl = float("inf")
    best_epoch = 0

    loss_history: List[float] = []

    global_batches = 0
    global_step = 0

    total_steps = epochs * len(train_loader)

    for epoch in range(1, epochs + 1):
        print(f"\nTraining epoch {epoch}/{epochs}...", flush=True)

        mean_loss, time_s, epoch_losses, global_batches, global_step = _train_epoch(
            model=model,
            opt=opt,
            train_loader=train_loader,
            epoch=epoch,
            epochs=epochs,
            tokenizer=tok,
            global_batch_offset=global_batches,
            global_step_offset=global_step,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            device=device,
        )
        loss_history.extend(epoch_losses)
        best_loss = min(best_loss, mean_loss)

        ppl = _eval_perplexity(model, val_loader, device=device)
        if ppl < best_ppl:
            best_ppl = ppl
            best_epoch = epoch

        ckpt_path = os.path.join("checkpoints", f"mink_large_epoch{epoch}.pt")
        torch.save(
            {
                "model_type": "mink",
                "config": cfg,
                "state_dict": model.state_dict(),
                "encoding_name": tok.encoding_name,
                "epoch": epoch,
                "best_loss": best_loss,
            },
            ckpt_path,
        )

        _ascii_loss_curve(loss_history, title="Minkowski")

        bar = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print(bar, flush=True)
        print(f"Epoch {epoch}/{epochs} complete", flush=True)
        print(
            f"mean_loss={mean_loss:.4f} | best_loss={best_loss:.4f} | time={_format_wall(time_s)}",
            flush=True,
        )
        print(f"Val perplexity: {ppl:.3f}", flush=True)
        print(bar, flush=True)
        print(f"Saved checkpoint: {ckpt_path}", flush=True)

    print("\nTRAINING COMPLETE", flush=True)
    print(f"Best perplexity: {best_ppl:.2f} (epoch {best_epoch})", flush=True)

    g_coh, w_coh = _mean_sequence_coherence(model, val_loader, tok, device=device)
    print(f"Wave vs Greedy coherence on val set: {w_coh:.3f} vs {g_coh:.3f}", flush=True)


if __name__ == "__main__":
    main()

