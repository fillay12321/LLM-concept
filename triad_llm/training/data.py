import torch


def generate_random_token_sequences(
    n: int,
    seq_len: int,
    vocab_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(0, vocab_size, (n, seq_len), device=device)
