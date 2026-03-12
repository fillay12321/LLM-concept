import matplotlib.pyplot as plt
import torch

from triad_llm.attention import MinkowskiAttention, StandardMultiheadAttention


def _percent_zeros(attn_weights: torch.Tensor) -> float:
    total = attn_weights.numel()
    zeros = (attn_weights == 0).sum().item()
    return 100.0 * zeros / float(total)


def _save_heatmap(attn: torch.Tensor, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(attn, cmap="Blues", vmin=0.0, vmax=attn.max().item())
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_xticks(torch.arange(attn.shape[1]))
    ax.set_yticks(torch.arange(attn.shape[0]))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _save_side_by_side(mink: torch.Tensor, std: torch.Tensor, filename: str) -> None:
    vmax = max(mink.max().item(), std.max().item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax1.imshow(mink, cmap="Blues", vmin=0.0, vmax=vmax)
    ax1.set_xlabel("Key tokens")
    ax1.set_ylabel("Query tokens")
    ax1.set_xticks(torch.arange(mink.shape[1]))
    ax1.set_yticks(torch.arange(mink.shape[0]))
    ax1.set_title("Minkowski Attention")

    im2 = ax2.imshow(std, cmap="Blues", vmin=0.0, vmax=vmax)
    ax2.set_xlabel("Key tokens")
    ax2.set_ylabel("Query tokens")
    ax2.set_xticks(torch.arange(std.shape[1]))
    ax2.set_yticks(torch.arange(std.shape[0]))
    ax2.set_title("Standard Attention")

    fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(42)

    seq_len = 16
    vocab_size = 50
    d_model = 128
    num_heads = 4

    device = torch.device("cpu")

    token_ids = torch.randint(0, vocab_size, (seq_len,), device=device)
    _ = token_ids

    embed = torch.randn(seq_len, 1, d_model, device=device)

    mink_attn = MinkowskiAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False, dropout=0.0).to(device)
    std_attn = StandardMultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False, dropout=0.0).to(device)

    _, mink_weights = mink_attn(embed, embed, embed, need_weights=True, average_attn_weights=True)
    _, std_weights = std_attn(embed, embed, embed, need_weights=True, average_attn_weights=True)

    mink_mat = mink_weights[0].cpu().detach()
    std_mat = std_weights[0].cpu().detach()

    mink_sparsity = _percent_zeros(mink_mat)

    _save_heatmap(
        mink_mat,
        title=f"Minkowski Attention (sparsity={mink_sparsity:.1f}%)",
        filename="minkowski_attention.png",
    )
    _save_heatmap(
        std_mat,
        title="Standard Attention",
        filename="standard_attention.png",
    )
    _save_side_by_side(mink_mat, std_mat, filename="comparison.png")

    print("Saved minkowski_attention.png, standard_attention.png, comparison.png")


if __name__ == "__main__":
    main()
