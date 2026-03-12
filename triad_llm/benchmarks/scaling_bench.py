import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

if __package__ is None or __package__ == "":
    bench_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.realpath(os.path.join(bench_dir, ".."))  # .../triad_llm
    repo_root = os.path.dirname(pkg_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from triad_llm.decoding import GreedyDecoder, WaveCollapseDecoder
from triad_llm.decoding.wave_collapse import wave_collapse_step_stats
from triad_llm.model import MinkowskiTransformer


def repetition_rate(seq: List[int]) -> float:
    if len(seq) <= 1:
        return 0.0
    rep = 0
    for i in range(len(seq) - 1):
        if seq[i] == seq[i + 1]:
            rep += 1
    return float(rep) / float(len(seq) - 1)


def ngram_repetition(seq: List[int], n: int = 3) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(seq) < n:
        return 0.0

    counts: Dict[Tuple[int, ...], int] = {}
    total = len(seq) - n + 1
    for i in range(total):
        g = tuple(seq[i : i + n])
        counts[g] = counts.get(g, 0) + 1

    repeated = 0
    for c in counts.values():
        if c > 1:
            repeated += c - 1

    return float(repeated) / float(total)


def energy(A: torch.Tensor) -> float:
    # E = -max(W) (depth of the minimum). We approximate decisiveness from the
    # collapsed amplitude distribution A. More decisive => larger max(A).
    return -float(A.to(dtype=torch.float32).max().item())


def _pairwise_mean_cosine(emb: torch.Tensor) -> float:
    # emb: (T, D)
    emb = F.normalize(emb.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)
    sim = emb @ emb.t()
    t = sim.shape[0]
    if t <= 1:
        return 1.0
    mask = ~torch.eye(t, dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


def _mean_entropy_from_probs(p: torch.Tensor) -> float:
    p = p.to(dtype=torch.float32).clamp_min(1e-12)
    return float((-(p * p.log()).sum()).item())


def _format_table(headers: List[str], rows: List[List[object]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))

    def fmt_row(r):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))

    out = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


@dataclass
class DecodeMetrics:
    entropy: float
    cos_sim: float
    repetition: float
    ngram_rep: float
    time_ms: float
    mean_iters_to_conv: float = 0.0
    mean_energy: float = 0.0
    wave_collapse_diversity: float = 0.0  # mean std(A)


@torch.no_grad()
def _eval_wave(
    model: MinkowskiTransformer,
    seeds: torch.Tensor,
    *,
    max_new_tokens: int,
    K: int,
    T: int,
    lambda_interference: float,
    gamma_context: float,
    tau: float,
) -> DecodeMetrics:
    entropies = []
    coherences = []
    reps = []
    nreps = []
    times = []
    iters = []
    energies = []
    a_stds = []

    for i in range(seeds.shape[0]):
        tokens = seeds[i : i + 1].clone()
        step_ent = []
        step_iters = []
        step_E = []
        step_std = []

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_id, final_A, iters_to_conv, _ = wave_collapse_step_stats(
                model,
                tokens,
                K=K,
                T=T,
                lambda_interference=lambda_interference,
                tau=tau,
                gamma_context=gamma_context,
                mu_diversity=0.0,
            )
            step_ent.append(_mean_entropy_from_probs(final_A))
            step_iters.append(float(iters_to_conv))
            step_E.append(energy(final_A))
            step_std.append(float(final_A.to(dtype=torch.float32).std(unbiased=False).item()))

            tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=tokens.dtype)], dim=1)
        t1 = time.perf_counter()

        seq = tokens[0].tolist()

        entropies.append(float(sum(step_ent) / max(len(step_ent), 1)))
        iters.append(float(sum(step_iters) / max(len(step_iters), 1)))
        energies.append(float(sum(step_E) / max(len(step_E), 1)))
        a_stds.append(float(sum(step_std) / max(len(step_std), 1)))
        times.append((t1 - t0) * 1000.0)

        emb = model.tok_emb(torch.tensor(seq, dtype=torch.long))
        coherences.append(_pairwise_mean_cosine(emb))
        reps.append(repetition_rate(seq))
        nreps.append(ngram_repetition(seq, n=3))

    return DecodeMetrics(
        entropy=float(sum(entropies) / len(entropies)),
        cos_sim=float(sum(coherences) / len(coherences)),
        repetition=float(sum(reps) / len(reps)),
        ngram_rep=float(sum(nreps) / len(nreps)),
        time_ms=float(sum(times) / len(times)),
        mean_iters_to_conv=float(sum(iters) / len(iters)),
        mean_energy=float(sum(energies) / len(energies)),
        wave_collapse_diversity=float(sum(a_stds) / len(a_stds)),
    )


@torch.no_grad()
def _eval_greedy(model: MinkowskiTransformer, seeds: torch.Tensor, *, max_new_tokens: int) -> DecodeMetrics:
    dec = GreedyDecoder(model)
    entropies = []
    coherences = []
    reps = []
    nreps = []
    times = []

    for i in range(seeds.shape[0]):
        seed = seeds[i : i + 1]
        tokens = seed.clone()
        step_ent = []

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = model(tokens)[0, -1]
            p = F.softmax(logits.to(dtype=torch.float32), dim=-1)
            step_ent.append(_mean_entropy_from_probs(p))
            next_id = int(logits.argmax(dim=-1).item())
            tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=tokens.dtype)], dim=1)
        seq = tokens[0].tolist()
        t1 = time.perf_counter()

        entropies.append(float(sum(step_ent) / max(len(step_ent), 1)))
        times.append((t1 - t0) * 1000.0)

        emb = model.tok_emb(torch.tensor(seq, dtype=torch.long))
        coherences.append(_pairwise_mean_cosine(emb))
        reps.append(repetition_rate(seq))
        nreps.append(ngram_repetition(seq, n=3))

    return DecodeMetrics(
        entropy=float(sum(entropies) / len(entropies)),
        cos_sim=float(sum(coherences) / len(coherences)),
        repetition=float(sum(reps) / len(reps)),
        ngram_rep=float(sum(nreps) / len(nreps)),
        time_ms=float(sum(times) / len(times)),
    )


@torch.no_grad()
def _eval_topk_sampling(
    model: MinkowskiTransformer,
    seeds: torch.Tensor,
    *,
    max_new_tokens: int,
    K: int,
) -> DecodeMetrics:
    entropies = []
    coherences = []
    reps = []
    nreps = []
    times = []

    for i in range(seeds.shape[0]):
        tokens = seeds[i : i + 1].clone()
        step_ent = []

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = model(tokens)[0, -1]
            p = F.softmax(logits.to(dtype=torch.float32), dim=-1)
            top_p, top_idx = torch.topk(p, k=min(int(K), p.numel()), dim=-1)
            top_p = top_p / top_p.sum().clamp_min(1e-12)

            step_ent.append(_mean_entropy_from_probs(top_p))

            sampled = int(top_idx[torch.multinomial(top_p, num_samples=1).item()].item())
            tokens = torch.cat([tokens, torch.tensor([[sampled]], dtype=tokens.dtype)], dim=1)

        seq = tokens[0].tolist()
        t1 = time.perf_counter()

        entropies.append(float(sum(step_ent) / max(len(step_ent), 1)))
        times.append((t1 - t0) * 1000.0)

        emb = model.tok_emb(torch.tensor(seq, dtype=torch.long))
        coherences.append(_pairwise_mean_cosine(emb))
        reps.append(repetition_rate(seq))
        nreps.append(ngram_repetition(seq, n=3))

    return DecodeMetrics(
        entropy=float(sum(entropies) / len(entropies)),
        cos_sim=float(sum(coherences) / len(coherences)),
        repetition=float(sum(reps) / len(reps)),
        ngram_rep=float(sum(nreps) / len(nreps)),
        time_ms=float(sum(times) / len(times)),
    )


def _score_best(m: DecodeMetrics) -> float:
    # Higher coherence, lower repetition.
    return float(m.cos_sim) - float(m.repetition) - float(m.ngram_rep)


def main():
    torch.manual_seed(42)

    base_cfg = {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "vocab_size": 500,
        "max_seq_len": 64,
        "dropout": 0.0,
    }

    device = torch.device("cpu")

    num_seqs = 50
    total_len = 32
    seed_len = 4
    max_new_tokens = total_len - seed_len

    seeds = torch.randint(0, base_cfg["vocab_size"], (num_seqs, seed_len), device=device)

    model = MinkowskiTransformer(**base_cfg).to(device)
    model.eval()

    print("\nEXPERIMENT 1: K scaling")
    K_values = [2, 4, 8, 16, 32]
    rows = []
    exp1: Dict[int, DecodeMetrics] = {}
    for K in K_values:
        m = _eval_wave(model, seeds, max_new_tokens=max_new_tokens, K=K, T=5, lambda_interference=0.3, gamma_context=0.2, tau=0.5)
        exp1[K] = m
        rows.append([K, f"{m.entropy:.4f}", f"{m.cos_sim:.4f}", f"{m.repetition:.4f}", f"{m.ngram_rep:.4f}", f"{m.time_ms:.2f}"])
    print(_format_table(["K", "entropy", "cos_sim", "rep", "ngram_rep", "time_ms"], rows))

    print("\nEXPERIMENT 2: Lambda scaling")
    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.2]
    rows = []
    exp2: Dict[float, DecodeMetrics] = {}
    for lam in lambda_values:
        m = _eval_wave(model, seeds, max_new_tokens=max_new_tokens, K=10, T=5, lambda_interference=lam, gamma_context=0.2, tau=0.5)
        exp2[lam] = m
        rows.append([lam, f"{m.entropy:.4f}", f"{m.cos_sim:.4f}", f"{m.repetition:.4f}", f"{m.ngram_rep:.4f}", f"{m.time_ms:.2f}"])
    print(_format_table(["lambda", "entropy", "cos_sim", "rep", "ngram_rep", "time_ms"], rows))

    print("\nEXPERIMENT 3: Gamma scaling")
    gamma_values = [0.0, 0.2, 0.5, 1.0, 2.0]
    rows = []
    exp3: Dict[float, DecodeMetrics] = {}
    for gam in gamma_values:
        m = _eval_wave(model, seeds, max_new_tokens=max_new_tokens, K=10, T=5, lambda_interference=0.3, gamma_context=gam, tau=0.5)
        exp3[gam] = m
        rows.append([gam, f"{m.entropy:.4f}", f"{m.cos_sim:.4f}", f"{m.repetition:.4f}", f"{m.ngram_rep:.4f}", f"{m.time_ms:.2f}"])
    print(_format_table(["gamma", "entropy", "cos_sim", "rep", "ngram_rep", "time_ms"], rows))

    print("\nEXPERIMENT 4: T (iterations) scaling")
    T_values = [1, 2, 4, 8]
    rows = []
    exp4: Dict[int, DecodeMetrics] = {}
    for T in T_values:
        m = _eval_wave(model, seeds, max_new_tokens=max_new_tokens, K=10, T=T, lambda_interference=0.3, gamma_context=0.2, tau=0.5)
        exp4[T] = m
        rows.append([T, f"{m.entropy:.4f}", f"{m.cos_sim:.4f}", f"{m.repetition:.4f}", f"{m.ngram_rep:.4f}", f"{m.time_ms:.2f}", f"{m.mean_iters_to_conv:.2f}"])
    print(_format_table(["T", "entropy", "cos_sim", "rep", "ngram_rep", "time_ms", "mean_iters_to_conv"], rows))

    # T=5 note
    m_T5 = _eval_wave(model, seeds, max_new_tokens=max_new_tokens, K=10, T=5, lambda_interference=0.3, gamma_context=0.2, tau=0.5)
    if 2 in exp4:
        rel = abs(exp4[2].cos_sim - m_T5.cos_sim) / max(abs(m_T5.cos_sim), 1e-12)
        if rel <= 0.05:
            print("Note: T=2 is within 5% of T=5 on cos_sim.")

    print("\nEXPERIMENT 5: Vocab size scaling")
    vocab_sizes = [50, 500, 2000]
    rows = []
    exp5: Dict[int, DecodeMetrics] = {}
    for vsz in vocab_sizes:
        mcfg = dict(base_cfg)
        mcfg["vocab_size"] = vsz
        m2 = MinkowskiTransformer(**mcfg).to(device)
        m2.eval()
        seeds2 = torch.randint(0, vsz, (num_seqs, seed_len), device=device)
        m = _eval_wave(m2, seeds2, max_new_tokens=max_new_tokens, K=10, T=5, lambda_interference=0.3, gamma_context=0.2, tau=0.5)
        exp5[vsz] = m
        rows.append([vsz, f"{m.entropy:.4f}", f"{m.cos_sim:.4f}", f"{m.repetition:.4f}", f"{m.ngram_rep:.4f}", f"{m.time_ms:.2f}"])
    print(_format_table(["vocab", "entropy", "cos_sim", "rep", "ngram_rep", "time_ms"], rows))

    print("\nEXPERIMENT 6: Decoder comparison")
    m_g = _eval_greedy(model, seeds, max_new_tokens=max_new_tokens)
    m_topk = _eval_topk_sampling(model, seeds, max_new_tokens=max_new_tokens, K=10)
    m_w = _eval_wave(model, seeds, max_new_tokens=max_new_tokens, K=10, T=5, lambda_interference=0.3, gamma_context=0.2, tau=0.5)

    rows = [
        ["Greedy", f"{m_g.entropy:.4f}", f"{m_g.cos_sim:.4f}", f"{m_g.repetition:.4f}", f"{m_g.ngram_rep:.4f}", "-", f"{m_g.time_ms:.2f}"],
        ["TopK", f"{m_topk.entropy:.4f}", f"{m_topk.cos_sim:.4f}", f"{m_topk.repetition:.4f}", f"{m_topk.ngram_rep:.4f}", "-", f"{m_topk.time_ms:.2f}"],
        ["Wave", f"{m_w.entropy:.4f}", f"{m_w.cos_sim:.4f}", f"{m_w.repetition:.4f}", f"{m_w.ngram_rep:.4f}", f"{m_w.mean_energy:.4f}", f"{m_w.time_ms:.2f}"],
    ]
    print(_format_table(["decoder", "entropy", "cos_sim", "rep", "ngram_rep", "mean_energy", "time_ms"], rows))

    # Summary
    best_K = max(exp1.items(), key=lambda kv: _score_best(kv[1]))[0]
    best_lambda = max(exp2.items(), key=lambda kv: _score_best(kv[1]))[0]

    max_cos_T = max(exp4.values(), key=lambda m: m.cos_sim).cos_sim
    min_T_95 = None
    for T in sorted(exp4.keys()):
        if exp4[T].cos_sim >= 0.95 * max_cos_T:
            min_T_95 = T
            break

    wave_beats_topk = m_w.cos_sim > m_topk.cos_sim

    print("\nSUMMARY")
    print(f"Best K value (coherence vs repetition tradeoff): {best_K}")
    print(f"Best lambda value: {best_lambda}")
    print(f"Minimum T achieving 95% of max cos_sim: {min_T_95}")
    print(f"Wave beats TopK sampling on coherence: {wave_beats_topk}")


if __name__ == "__main__":
    main()
