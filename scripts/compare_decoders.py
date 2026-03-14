import argparse
import time
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from triad_llm.decoding.hf_wave_collapse import HFWaveCollapseDecoder


def _pairwise_cos_sim(model, token_ids: List[int]) -> float:
    if len(token_ids) <= 1:
        return 1.0
    device = next(model.parameters()).device
    emb = model.get_input_embeddings()
    x = emb(torch.tensor(token_ids, dtype=torch.long, device=device))
    x = F.normalize(x.to(dtype=torch.float32), p=2, dim=-1, eps=1e-12)
    sim = x @ x.t()
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    return float(sim[mask].mean().item())


def _repetition_rate(token_ids: List[int]) -> float:
    if not token_ids:
        return 0.0
    n = len(token_ids)
    unique = len(set(token_ids))
    return float(1.0 - unique / max(n, 1))


def _entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    p = p.to(dtype=torch.float32).clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def _sequence_entropy(model, full_ids: List[int], gen_len: int) -> float:
    device = next(model.parameters()).device
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits  # (1, L, V)
        logits = logits[0, -gen_len:, :]  # (gen_len, V)
        probs = F.softmax(logits, dim=-1)
        ent = _entropy_from_probs(probs)  # (gen_len,)
    return float(ent.mean().item())


def _generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> Tuple[str, List[int], int]:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt_ms = int((time.perf_counter() - t0) * 1000)

    full_ids = out[0].tolist()
    gen_ids = full_ids[prompt_len:]
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return text, gen_ids, dt_ms


def _generate_topk(model, tokenizer, prompt: str, max_new_tokens: int, k: int = 50) -> Tuple[str, List[int], int]:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=k,
            top_p=1.0,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt_ms = int((time.perf_counter() - t0) * 1000)

    full_ids = out[0].tolist()
    gen_ids = full_ids[prompt_len:]
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return text, gen_ids, dt_ms


def _generate_wave(model, tokenizer, prompt: str, max_new_tokens: int) -> Tuple[str, List[int], int]:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    decoder = HFWaveCollapseDecoder(
        model,
        tokenizer,
        K=8,
        T=3,
        lambda_interference=0.1,
        gamma_context=1.0,
        tau=0.5,
        repetition_penalty=2.5,
    )

    t0 = time.perf_counter()
    text = decoder.generate(prompt, max_new_tokens=max_new_tokens)
    dt_ms = int((time.perf_counter() - t0) * 1000)

    # Re-tokenize to recover ids (decoder already used same tokenizer)
    full_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    gen_ids = full_ids[-max_new_tokens:]
    return text, gen_ids, dt_ms


def _metrics_for_sample(model, tokenizer, prompt: str, max_new_tokens: int) -> Dict[str, Dict[str, float]]:
    greedy_text, greedy_ids, greedy_ms = _generate_greedy(model, tokenizer, prompt, max_new_tokens)
    topk_text, topk_ids, topk_ms = _generate_topk(model, tokenizer, prompt, max_new_tokens, k=50)
    wave_text, wave_ids, wave_ms = _generate_wave(model, tokenizer, prompt, max_new_tokens)

    # For entropy we need full sequences; approximate by using prompt + continuation reconstructed via tokenizer
    greedy_full_ids = tokenizer(greedy_text, return_tensors="pt")["input_ids"][0].tolist()
    topk_full_ids = tokenizer(topk_text, return_tensors="pt")["input_ids"][0].tolist()
    wave_full_ids = tokenizer(wave_text, return_tensors="pt")["input_ids"][0].tolist()

    metrics = {
        "greedy": {
            "text": greedy_text,
            "coherence": _pairwise_cos_sim(model, greedy_ids),
            "repetition": _repetition_rate(greedy_ids),
            "entropy": _sequence_entropy(model, greedy_full_ids, len(greedy_ids)),
            "time_ms": float(greedy_ms),
        },
        "topk": {
            "text": topk_text,
            "coherence": _pairwise_cos_sim(model, topk_ids),
            "repetition": _repetition_rate(topk_ids),
            "entropy": _sequence_entropy(model, topk_full_ids, len(topk_ids)),
            "time_ms": float(topk_ms),
        },
        "wave": {
            "text": wave_text,
            "coherence": _pairwise_cos_sim(model, wave_ids),
            "repetition": _repetition_rate(wave_ids),
            "entropy": _sequence_entropy(model, wave_full_ids, len(wave_ids)),
            "time_ms": float(wave_ms),
        },
    }
    return metrics


def _print_single_prompt_results(prompt: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print("=" * 80)
    print(f"Prompt:\n{prompt}")
    print("-" * 80)
    for name in ["greedy", "topk", "wave"]:
        m = metrics[name]
        print(f"[{name.upper()}]")
        print(f"Time: {m['time_ms']:.0f} ms | Coherence: {m['coherence']:.3f} | "
              f"Repetition: {m['repetition']:.3f} | Entropy: {m['entropy']:.3f}")
        print(m["text"])
        print("-" * 80)


def _benchmark_wikitext(model, tokenizer, max_new_tokens: int, n_prompts: int = 50) -> None:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    prompts: List[str] = []
    for row in ds:
        text = row.get("text", "").strip()
        if text:
            prompts.append(text)
        if len(prompts) >= n_prompts:
            break

    agg: Dict[str, Dict[str, float]] = {
        "greedy": {"coherence": 0.0, "repetition": 0.0, "entropy": 0.0, "time_ms": 0.0},
        "topk": {"coherence": 0.0, "repetition": 0.0, "entropy": 0.0, "time_ms": 0.0},
        "wave": {"coherence": 0.0, "repetition": 0.0, "entropy": 0.0, "time_ms": 0.0},
    }

    for i, prompt in enumerate(prompts, start=1):
        print(f"Benchmarking prompt {i}/{len(prompts)}...", end="\r", flush=True)
        m = _metrics_for_sample(model, tokenizer, prompt, max_new_tokens)
        for name in ["greedy", "topk", "wave"]:
            agg[name]["coherence"] += m[name]["coherence"]
            agg[name]["repetition"] += m[name]["repetition"]
            agg[name]["entropy"] += m[name]["entropy"]
            agg[name]["time_ms"] += m[name]["time_ms"]

    print()
    print("=" * 80)
    print(f"Wikitext-2 benchmark over {len(prompts)} prompts (max_new_tokens={max_new_tokens})")
    print("-" * 80)
    header = f"{'Decoder':<10} | {'Coherence':>9} | {'Repetition':>11} | {'Entropy':>8} | {'Time ms':>8}"
    print(header)
    print("-" * len(header))

    for name in ["greedy", "topk", "wave"]:
        n = float(len(prompts))
        coh = agg[name]["coherence"] / n
        rep = agg[name]["repetition"] / n
        ent = agg[name]["entropy"] / n
        tms = agg[name]["time_ms"] / n
        print(f"{name:<10} | {coh:9.3f} | {rep:11.3f} | {ent:8.3f} | {tms:8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Number of new tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    metrics = _metrics_for_sample(model, tokenizer, args.prompt, args.max_new_tokens)
    _print_single_prompt_results(args.prompt, metrics)

    _benchmark_wikitext(model, tokenizer, args.max_new_tokens, n_prompts=50)


if __name__ == "__main__":
    main()

