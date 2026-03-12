import torch

from triad_llm.model import MinkowskiTransformer, StandardTransformer
from triad_llm.training import eval_perplexity, generate_random_token_sequences, train_language_model


def main():
    torch.manual_seed(42)

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

    train_seqs = generate_random_token_sequences(train_n, seq_len, vocab_size, device=device)
    val_seqs = generate_random_token_sequences(val_n, seq_len, vocab_size, device=device)

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

    mink_params = sum(p.numel() for p in mink.parameters())
    std_params = sum(p.numel() for p in std.parameters())
    print(f"Minkowski params: {mink_params:,}")
    print(f"Standard params: {std_params:,}")

    mink_losses, _ = train_language_model(mink, train_seqs, epochs, batch_size, lr)
    std_losses, _ = train_language_model(std, train_seqs, epochs, batch_size, lr)

    for i in range(epochs):
        print(f"Epoch {i + 1}: Minkowski loss={mink_losses[i]:.4f} | Standard loss={std_losses[i]:.4f}")

    mink_ppl = eval_perplexity(mink, val_seqs, batch_size=batch_size)
    std_ppl = eval_perplexity(std, val_seqs, batch_size=batch_size)

    print(f"Held-out perplexity: Minkowski={mink_ppl:.3f} | Standard={std_ppl:.3f}")


if __name__ == "__main__":
    main()
