# TRIAD-LLM-concept

## Description
Minkowski spacetime attention — a novel transformer attention mechanism based on special relativity. Causality emerges from spacetime geometry, not from artificial masks.

## Installation
```bash
pip install torch
```

For editable installs of this repo:
```bash
pip install -e .
```

## Quick start
```python
import torch
from triad_llm.attention import MinkowskiAttention

attn = MinkowskiAttention(embed_dim=128, num_heads=4, batch_first=False)

x = torch.randn(16, 2, 128)
out, w = attn(x, x, x, need_weights=True, average_attn_weights=True)
print(out.shape, w.shape)
```

## Benchmark results
- Attention sparsity: 52% vs 0% (Standard)
- Entropy improvement: -19%
- Perplexity on synthetic data: 50.103 vs 50.101
