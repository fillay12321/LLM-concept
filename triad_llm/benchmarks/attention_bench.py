import time

import torch

from triad_llm.attention import MinkowskiAttention, StandardMultiheadAttention


def _percent_zeros(attn_weights: torch.Tensor) -> float:
    total = attn_weights.numel()
    zeros = (attn_weights == 0).sum().item()
    return 100.0 * zeros / float(total)


def _mean_entropy(attn_weights: torch.Tensor, eps: float = 1e-12) -> float:
    if attn_weights.dim() == 4:
        attn = attn_weights.mean(dim=1)
    else:
        attn = attn_weights

    p = attn.to(dtype=torch.float32).clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)
    return ent.mean().item()


@torch.no_grad()
def _bench(module: torch.nn.Module, x: torch.Tensor, runs: int = 10, warmup: int = 3):
    module.eval()

    for _ in range(warmup):
        _ = module(x, x, x, need_weights=True, average_attn_weights=True)

    t0 = time.perf_counter()
    out = None
    w = None
    for _ in range(runs):
        out, w = module(x, x, x, need_weights=True, average_attn_weights=True)
    t1 = time.perf_counter()

    avg_time_s = (t1 - t0) / float(runs)

    return {
        "sparsity_pct": _percent_zeros(w),
        "entropy": _mean_entropy(w),
        "time_ms": avg_time_s * 1000.0,
        "out_norm": out.to(dtype=torch.float32).norm().item(),
    }


def main():
    torch.manual_seed(42)

    batch_size = 4
    d_model = 256
    num_heads = 8

    device = torch.device("cpu")

    configs = [64, 128, 256]

    mink = MinkowskiAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False, dropout=0.0).to(device)
    std = StandardMultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False, dropout=0.0).to(device)

    fmt = (
        "{seq:>4} | {mz:>5} {sz:>5} | {mH:>6} {sH:>6} | "
        "{mms:>7} {sms:>7} | {mo:>7} {so:>7}"
    )
    print(fmt.format(seq="seq", mz="Mz%", sz="Sz%", mH="MH", sH="SH", mms="Mms", sms="Sms", mo="Mnorm", so="Snorm"))
    print("-" * 79)

    ent_m = []
    ent_s = []

    for seq_len in configs:
        x = torch.randn(seq_len, batch_size, d_model, device=device)
        rm = _bench(mink, x)
        rs = _bench(std, x)

        ent_m.append(rm["entropy"])
        ent_s.append(rs["entropy"])

        print(
            fmt.format(
                seq=seq_len,
                mz=f"{rm['sparsity_pct']:.2f}",
                sz=f"{rs['sparsity_pct']:.2f}",
                mH=f"{rm['entropy']:.4f}",
                sH=f"{rs['entropy']:.4f}",
                mms=f"{rm['time_ms']:.3f}",
                sms=f"{rs['time_ms']:.3f}",
                mo=f"{rm['out_norm']:.3f}",
                so=f"{rs['out_norm']:.3f}",
            )
        )

    mean_ent_m = sum(ent_m) / float(len(ent_m))
    mean_ent_s = sum(ent_s) / float(len(ent_s))
    focus_improvement_pct = 100.0 * (mean_ent_s - mean_ent_m) / max(mean_ent_s, 1e-12)

    if mean_ent_m < mean_ent_s:
        print(
            f"Verdict: Minkowski more focused by {focus_improvement_pct:.2f}% "
            f"(mean H {mean_ent_m:.4f} vs {mean_ent_s:.4f})."
        )
    elif mean_ent_m > mean_ent_s:
        print(
            f"Verdict: Standard more focused by {abs(focus_improvement_pct):.2f}% "
            f"(mean H {mean_ent_s:.4f} vs {mean_ent_m:.4f})."
        )
    else:
        print(f"Verdict: tie (mean H {mean_ent_m:.4f}).")


if __name__ == "__main__":
    main()
