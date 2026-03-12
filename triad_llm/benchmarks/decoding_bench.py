import time
import os
import sys

import torch
from torch.nn import functional as F

if __package__ is None or __package__ == "":
    # Allow running as a script: `python triad_llm/benchmarks/decoding_bench.py`
    # from any working directory.
    bench_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.realpath(os.path.join(bench_dir, ".."))  # .../triad_llm
    repo_root = os.path.dirname(pkg_dir)  # parent of triad_llm
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from triad_llm.decoding import GreedyDecoder, WaveCollapseDecoder
from triad_llm.decoding.wave_collapse import wave_collapse_step_stats
from triad_llm.model import MinkowskiTransformer


def _pairwise_mean_cosine(emb: torch.Tensor) -> float:
    # emb: (T, D)
    emb = F.normalize(emb.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)
    sim = emb @ emb.t()  # (T, T)
    t = sim.shape[0]
    if t <= 1:
        return 1.0
    mask = ~torch.eye(t, dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


@torch.no_grad()
def _decode_stats(decoder_name: str, decoder, model: MinkowskiTransformer, seeds: torch.Tensor, max_new_tokens: int, K: int, lam: float):
    entropies = []
    coherences = []
    times = []
    examples = []

    device = torch.device("cpu")
    _ = device

    for i in range(seeds.shape[0]):
        seed = seeds[i : i + 1]

        t0 = time.perf_counter()

        if decoder_name == "wave":
            tokens = seed.clone()
            step_H = []
            for _ in range(max_new_tokens):
                next_id, entropy_W, _ = wave_collapse_step_stats(model, tokens, K=K, lambda_interference=lam)
                step_H.append(entropy_W)
                tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=tokens.dtype)], dim=1)
            seq = tokens[0].tolist()
            entropies.append(float(sum(step_H) / max(len(step_H), 1)))
        else:
            seq = decoder.generate(seed, max_new_tokens=max_new_tokens)
            # For greedy, compute entropy of the model distribution at each step for comparability.
            tokens = seed.clone()
            step_H = []
            for _ in range(max_new_tokens):
                logits = model(tokens)[0, -1]
                p = F.softmax(logits.to(dtype=torch.float32), dim=-1).clamp_min(1e-12)
                step_H.append(float((-(p * p.log()).sum()).item()))
                next_id = int(logits.argmax(dim=-1).item())
                tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=tokens.dtype)], dim=1)
            entropies.append(float(sum(step_H) / max(len(step_H), 1)))

        t1 = time.perf_counter()
        times.append((t1 - t0))

        seq_t = torch.tensor(seq, dtype=torch.long)
        emb = model.tok_emb(seq_t)
        coherences.append(_pairwise_mean_cosine(emb))

        if len(examples) < 5:
            examples.append(seq)

    return {
        "mean_entropy": float(sum(entropies) / len(entropies)),
        "mean_coherence": float(sum(coherences) / len(coherences)),
        "time_per_seq_s": float(sum(times) / len(times)),
        "examples": examples,
    }


def main():
    torch.manual_seed(42)

    d_model = 128
    num_heads = 4
    num_layers = 4
    vocab_size = 50
    max_seq_len = 64

    num_seqs = 100
    seed_len = 4
    max_new_tokens = 32

    K = 5
    lam = 0.3

    device = torch.device("cpu")

    model = MinkowskiTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0,
    ).to(device)
    model.eval()

    seeds = torch.randint(0, vocab_size, (num_seqs, seed_len), device=device)

    greedy = GreedyDecoder(model)
    wave = WaveCollapseDecoder(model, K=K, lambda_interference=lam)

    res_g = _decode_stats("greedy", greedy, model, seeds, max_new_tokens=max_new_tokens, K=K, lam=lam)
    res_w = _decode_stats("wave", wave, model, seeds, max_new_tokens=max_new_tokens, K=K, lam=lam)

    print("Decoder | mean_entropy | mean_cos_sim | time_per_seq(ms)")
    print("-" * 54)
    print(
        f"Greedy  | {res_g['mean_entropy']:.4f}      | {res_g['mean_coherence']:.4f}      | {res_g['time_per_seq_s']*1000.0:.2f}"
    )
    print(
        f"Wave    | {res_w['mean_entropy']:.4f}      | {res_w['mean_coherence']:.4f}      | {res_w['time_per_seq_s']*1000.0:.2f}"
    )

    print("\nExamples (Greedy vs Wave):")
    for i in range(5):
        g = res_g["examples"][i]
        w = res_w["examples"][i]
        print(f"{i+1:>2}. G: {g}")
        print(f"    W: {w}")


if __name__ == "__main__":
    main()
