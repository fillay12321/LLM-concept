import argparse

import torch
from torch.nn import functional as F

from triad_llm.decoding import GreedyDecoder, WaveCollapseDecoder
from triad_llm.model import MinkowskiTransformer, StandardTransformer
from triad_llm.training.tokenizer import TiktokenWrapper


@torch.no_grad()
def topk_generate(model, seed_tokens: torch.Tensor, *, max_new_tokens: int, K: int = 10) -> list[int]:
    tokens = seed_tokens.clone()
    for _ in range(max_new_tokens):
        logits = model(tokens)[0, -1]
        p = F.softmax(logits.to(dtype=torch.float32), dim=-1)
        top_p, top_idx = torch.topk(p, k=min(int(K), p.numel()), dim=-1)
        top_p = top_p / top_p.sum().clamp_min(1e-12)
        sampled = int(top_idx[torch.multinomial(top_p, num_samples=1).item()].item())
        tokens = torch.cat([tokens, torch.tensor([[sampled]], dtype=tokens.dtype)], dim=1)
    return tokens[0].tolist()


def _load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    mtype = ckpt.get("model_type", "mink")

    if mtype == "std":
        model = StandardTransformer(**cfg)
    else:
        model = MinkowskiTransformer(**cfg)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, cfg, mtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    torch.manual_seed(42)

    tok = TiktokenWrapper()
    model, cfg, mtype = _load_model(args.checkpoint)

    prompt_ids = tok.encode(args.prompt)
    max_seq_len = int(cfg["max_seq_len"])
    if len(prompt_ids) > max_seq_len:
        prompt_ids = prompt_ids[-max_seq_len:]

    seed = torch.tensor([prompt_ids], dtype=torch.long)

    greedy = GreedyDecoder(model)
    wave = WaveCollapseDecoder(model, K=10, T=5, lambda_interference=0.3, tau=0.5, gamma_context=0.2, mu_diversity=0.0)

    greedy_ids = greedy.generate(seed, max_new_tokens=200)
    topk_ids = topk_generate(model, seed, max_new_tokens=200, K=10)
    wave_ids = wave.generate(seed, max_new_tokens=200)

    print(f"Loaded checkpoint type: {mtype}")

    print("GREEDY:")
    print(tok.decode(greedy_ids))
    print("\nTOPK:")
    print(tok.decode(topk_ids))
    print("\nWAVE:")
    print(tok.decode(wave_ids))


if __name__ == "__main__":
    main()
