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
    wave_iters = []
    wave_a_entropy = []
    wave_diversity = []

    device = torch.device("cpu")
    _ = device

    for i in range(seeds.shape[0]):
        seed = seeds[i : i + 1]

        t0 = time.perf_counter()

        if decoder_name == "wave":
            tokens = seed.clone()
            step_H = []
            step_iters = []
            step_a_ent = []
            step_a_std = []
            for _ in range(max_new_tokens):
                next_id, final_A, iters_to_conv, _ = wave_collapse_step_stats(
                    model,
                    tokens,
                    K=K,
                    T=5,
                    lambda_interference=lam,
                    tau=0.5,
                    gamma_context=0.2,
                    mu_diversity=0.1,
                )
                step_H.append(float((-(final_A.clamp_min(1e-12) * final_A.clamp_min(1e-12).log()).sum()).item()))
                step_iters.append(int(iters_to_conv))
                step_a_ent.append(float((-(final_A.clamp_min(1e-12) * final_A.clamp_min(1e-12).log()).sum()).item()))
                step_a_std.append(float(final_A.to(dtype=torch.float32).std(unbiased=False).item()))
                tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=tokens.dtype)], dim=1)
            seq = tokens[0].tolist()
            entropies.append(float(sum(step_H) / max(len(step_H), 1)))
            wave_iters.append(float(sum(step_iters) / max(len(step_iters), 1)))
            wave_a_entropy.append(float(sum(step_a_ent) / max(len(step_a_ent), 1)))
            wave_diversity.append(float(sum(step_a_std) / max(len(step_a_std), 1)))
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
        "mean_wave_iters": float(sum(wave_iters) / len(wave_iters)) if len(wave_iters) > 0 else 0.0,
        "mean_wave_a_entropy": float(sum(wave_a_entropy) / len(wave_a_entropy)) if len(wave_a_entropy) > 0 else 0.0,
        "mean_wave_diversity": float(sum(wave_diversity) / len(wave_diversity)) if len(wave_diversity) > 0 else 0.0,
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

    K = 10
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
    wave = WaveCollapseDecoder(model, K=K, T=5, lambda_interference=lam, tau=0.5, gamma_context=0.2, mu_diversity=0.1)

    res_g = _decode_stats("greedy", greedy, model, seeds, max_new_tokens=max_new_tokens, K=K, lam=lam)
    res_w = _decode_stats("wave", wave, model, seeds, max_new_tokens=max_new_tokens, K=K, lam=lam)

    print("Decoder | mean_entropy | mean_cos_sim | time_per_seq(ms) | wave_iters | A_entropy | wave_collapse_diversity")
    print("-" * 86)
    print(
        f"Greedy  | {res_g['mean_entropy']:.4f}      | {res_g['mean_coherence']:.4f}      | {res_g['time_per_seq_s']*1000.0:.2f}          | {0.0:.2f}     | {0.0:.4f}   | {0.0:.4f}"
    )
    print(
        f"Wave    | {res_w['mean_entropy']:.4f}      | {res_w['mean_coherence']:.4f}      | {res_w['time_per_seq_s']*1000.0:.2f}          | {res_w['mean_wave_iters']:.2f}     | {res_w['mean_wave_a_entropy']:.4f}   | {res_w['mean_wave_diversity']:.4f}"
    )

    print("\nExamples (Greedy vs Wave):")
    for i in range(5):
        g = res_g["examples"][i]
        w = res_w["examples"][i]
        print(f"{i+1:>2}. G: {g}")
        print(f"    W: {w}")


if __name__ == "__main__":
    main()
