import time

import torch

from minkowski_attention import MinkowskiAttention, StandardMultiheadAttention


def _percent_zeros(attn_weights: torch.Tensor) -> float:
    # Count exact zeros for visualization / sparsity comparison.
    # attn_weights expected float tensor.
    total = attn_weights.numel()
    zeros = (attn_weights == 0).sum().item()
    return 100.0 * zeros / float(total)


@torch.no_grad()
def _bench(module: torch.nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 50):
    module.eval()

    # Warm-up
    for _ in range(warmup):
        _ = module(x, x, x, need_weights=True)

    t0 = time.perf_counter()
    out = None
    w = None
    for _ in range(iters):
        out, w = module(x, x, x, need_weights=True)
    t1 = time.perf_counter()

    return out, w, (t1 - t0) / float(iters)


def main():
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 64
    d_model = 256
    num_heads = 8

    device = torch.device("cpu")

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    mink = MinkowskiAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=True,
        dropout=0.0,
    ).to(device)

    std = StandardMultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=True,
        dropout=0.0,
    ).to(device)

    _, w_mink, t_mink = _bench(mink, x)
    _, w_std, t_std = _bench(std, x)

    # w_mink/w_std shapes: (N, L, S) (since average_attn_weights=True by default)
    z_mink = _percent_zeros(w_mink)
    z_std = _percent_zeros(w_std)

    print(f"Minkowski attention: {z_mink:.2f}% zero weights")
    print(f"Standard attention : {z_std:.2f}% zero weights")
    print(f"Minkowski inference time (avg): {t_mink * 1000.0:.3f} ms")
    print(f"Standard inference time (avg) : {t_std * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
