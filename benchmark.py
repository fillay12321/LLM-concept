import time

import torch

from minkowski_attention import MinkowskiAttention, StandardMultiheadAttention


def _percent_zeros(attn_weights: torch.Tensor) -> float:
    total = attn_weights.numel()
    zeros = (attn_weights == 0).sum().item()
    return 100.0 * zeros / float(total)


def _mean_entropy(attn_weights: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean entropy over batch and target positions.

    Expects weights shaped either:
    - (N, L, S)  (averaged over heads)
    - (N, H, L, S)

    Returns scalar python float.
    """
    if attn_weights.dim() == 4:
        # Average over heads to get a per-token distribution.
        attn = attn_weights.mean(dim=1)
    else:
        attn = attn_weights

    # attn: (N, L, S)
    p = attn.to(dtype=torch.float32).clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)  # (N, L)
    return ent.mean().item()


@torch.no_grad()
def _bench(
    module: torch.nn.Module,
    x: torch.Tensor,
    runs: int = 10,
    warmup: int = 3,
):
    module.eval()

    # Warmup
    for _ in range(warmup):
        _ = module(x, x, x, need_weights=True, average_attn_weights=True)

    # Timed runs
    t0 = time.perf_counter()
    out = None
    w = None
    for _ in range(runs):
        out, w = module(x, x, x, need_weights=True, average_attn_weights=True)
    t1 = time.perf_counter()

    avg_time_s = (t1 - t0) / float(runs)

    sparsity = _percent_zeros(w)
    entropy = _mean_entropy(w)
    out_norm = out.to(dtype=torch.float32).norm().item()

    return {
        "sparsity_pct": sparsity,
        "entropy": entropy,
        "time_ms": avg_time_s * 1000.0,
        "out_norm": out_norm,
    }


def _fmt_row(cols, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def main():
    torch.manual_seed(42)

    batch_size = 4
    d_model = 256
    num_heads = 8

    device = torch.device("cpu")

    configs = [
        (64, batch_size, d_model, num_heads),
        (128, batch_size, d_model, num_heads),
        (256, batch_size, d_model, num_heads),
    ]

    mink = MinkowskiAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=False,
        dropout=0.0,
    ).to(device)

    std = StandardMultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=False,
        dropout=0.0,
    ).to(device)

    rows = []

    # Track focus advantage via entropy (lower is more focused)
    ent_mink = []
    ent_std = []

    for (seq_len, bsz, d, h) in configs:
        x = torch.randn(seq_len, bsz, d, device=device)

        res_m = _bench(mink, x)
        res_s = _bench(std, x)

        ent_mink.append(res_m["entropy"])
        ent_std.append(res_s["entropy"])

        rows.append(
            {
                "seq_len": seq_len,
                "mink": res_m,
                "std": res_s,
            }
        )

    headers = [
        "seq_len",
        "Mink z%",
        "Std z%",
        "Mink H",
        "Std H",
        "Mink ms",
        "Std ms",
        "Mink ||o||",
        "Std ||o||",
    ]

    table = []
    table.append(headers)
    for r in rows:
        seq_len = r["seq_len"]
        m = r["mink"]
        s = r["std"]
        table.append(
            [
                seq_len,
                f"{m['sparsity_pct']:.2f}",
                f"{s['sparsity_pct']:.2f}",
                f"{m['entropy']:.4f}",
                f"{s['entropy']:.4f}",
                f"{m['time_ms']:.3f}",
                f"{s['time_ms']:.3f}",
                f"{m['out_norm']:.3f}",
                f"{s['out_norm']:.3f}",
            ]
        )

    # Print a compact fixed-width table (keep < ~80 chars for narrow terminals)
    fmt = (
        "{seq:>4} | {mz:>5} {sz:>5} | {mH:>6} {sH:>6} | "
        "{mms:>7} {sms:>7} | {mo:>7} {so:>7}"
    )
    print(
        fmt.format(
            seq="seq",
            mz="Mz%",
            sz="Sz%",
            mH="MH",
            sH="SH",
            mms="Mms",
            sms="Sms",
            mo="Mnorm",
            so="Snorm",
        )
    )
    print("-" * 79)
    for row in table[1:]:
        print(
            fmt.format(
                seq=row[0],
                mz=row[1],
                sz=row[2],
                mH=row[3],
                sH=row[4],
                mms=row[5],
                sms=row[6],
                mo=row[7],
                so=row[8],
            )
        )

    mean_ent_m = sum(ent_mink) / float(len(ent_mink))
    mean_ent_s = sum(ent_std) / float(len(ent_std))

    # Positive means Minkowski is more focused (lower entropy)
    focus_improvement_pct = 100.0 * (mean_ent_s - mean_ent_m) / max(mean_ent_s, 1e-12)

    if mean_ent_m < mean_ent_s:
        verdict = (
            f"Verdict: Minkowski more focused by {focus_improvement_pct:.2f}% "
            f"(mean H {mean_ent_m:.4f} vs {mean_ent_s:.4f})."
        )
    elif mean_ent_m > mean_ent_s:
        verdict = (
            f"Verdict: Standard more focused by {abs(focus_improvement_pct):.2f}% "
            f"(mean H {mean_ent_s:.4f} vs {mean_ent_m:.4f})."
        )
    else:
        verdict = f"Verdict: tie (mean H {mean_ent_m:.4f})."

    print(verdict)


if __name__ == "__main__":
    main()
