import torch

from triad_llm.model import MinkowskiTransformer, StandardTransformer


def _run_activation_isolation(model: torch.nn.Module, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        logits_a = model(tokens_a)
        logits_b = model(tokens_b)

    # If causal masking is correct, changing the last token must not affect logits at earlier positions.
    torch.testing.assert_close(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=0.0, rtol=0.0)


def _run_gradient_isolation(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    # Loss excludes last position; with correct causal masking, the last token should have no influence.
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = logits[:, :-1, :].sum()
    loss.backward()

    last_id = int(tokens[0, -1].item())
    grad = model.tok_emb.weight.grad[last_id]
    assert grad is not None
    assert float(grad.abs().max().item()) == 0.0


def test_standard_transformer_causal_isolation() -> None:
    torch.manual_seed(0)
    vocab_size = 97
    max_seq_len = 16
    d_model = 32
    num_heads = 4
    num_layers = 2
    dropout = 0.0

    model = StandardTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    tokens_a = torch.randint(0, vocab_size, (1, 8), dtype=torch.long)
    tokens_b = tokens_a.clone()
    tokens_b[0, -1] = (tokens_b[0, -1] + 1) % vocab_size

    _run_activation_isolation(model, tokens_a, tokens_b)
    _run_gradient_isolation(model, tokens_a)


def test_minkowski_transformer_causal_isolation() -> None:
    torch.manual_seed(0)
    vocab_size = 97
    max_seq_len = 16
    d_model = 32
    num_heads = 4
    num_layers = 2
    dropout = 0.0

    model = MinkowskiTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    tokens_a = torch.randint(0, vocab_size, (1, 8), dtype=torch.long)
    tokens_b = tokens_a.clone()
    tokens_b[0, -1] = (tokens_b[0, -1] + 1) % vocab_size

    _run_activation_isolation(model, tokens_a, tokens_b)
    _run_gradient_isolation(model, tokens_a)

