# my-llm

An LLM inference engine built from scratch on [MLX](https://github.com/ml-explore/mlx),
using low-level array/matrix APIs only — no high-level neural network abstractions.

Started as my solutions to the [tiny-llm course](https://github.com/skyzh/tiny-llm),
this repository is now developed independently.

## Current status

- **Week 1 — model basics:** attention (incl. GQA), RoPE, RMSNorm/MLP, Qwen3 model,
  generation, sampling — done
- **Week 2 — serving optimizations:** KV cache, quantized matmul, flash attention —
  in progress

## Layout

- `src/tiny_llm/` — Python implementation (model, generation, serving)
- `src/extensions/` — C++/Metal kernels exposed via nanobind (`tiny_llm_ext`)
- `tests/` — per-day test suites
- `main.py`, `bench.py`, `batch-main.py` — run & benchmark entry points

## Development

```bash
pdm install -v
pdm run check-installation
pdm run build-ext                     # build C++/Metal extensions
pdm run test tests/test_week_1_day_1.py
pdm run main-week1                    # run the week-1 model
```

## Origin

The project baseline is derived from the tiny-llm course starter code,
licensed under Apache-2.0 (see `LICENSE`). Course scaffolding (book,
reference solutions) has been removed; this repository stands on its own.
