# Boltz inference A/B benchmark harness

`boltz_bench.py` is a standalone CLI that folds a small set of inputs under a
**baseline** and an **experimental** `boltz predict` configuration and reports
the deltas in wall time, peak GPU memory, and confidence metrics. It is built
for high-throughput antibody-vs-fixed-antigen folding where you consume the CIF
and the `confidence_*.json` outputs.

> **Wall-clock and VRAM numbers require a GPU box** (A100 / L40S / A6000 /
> B200). All parsing, aggregation, A/B-delta, CSV/summary, and RMSD logic is
> GPU-free and unit tested (`tests/bench/test_boltz_bench.py`).

## What it does

1. Runs `boltz predict <data> <extra-args> --out_dir <tmp>` as a subprocess for
   each config, `--repeat N` times (wall time averaged).
2. Captures **wall time** (total, plus an even-split per-target proxy) and
   **peak GPU memory** by polling `nvidia-smi --query-gpu=memory.used` in a
   background thread. If a run drops a `torch_max_memory_allocated.txt` marker
   file in its out dir, the `torch.cuda.max_memory_allocated` value is also
   reported. Both are optional: missing `nvidia-smi` records `None`.
3. Parses every `confidence_*.json` and extracts the scalar metrics
   (`confidence_score`, `ptm`, `iptm`, `complex_plddt`, `complex_iplddt`,
   `complex_pde`, `complex_ipde`) per `(target, model)`, then aggregates per
   target (mean across models).
4. Produces an A/B comparison per target: wall-time speedup
   (`baseline / exp`), peak-VRAM delta (`exp − baseline`), and per-metric mean
   deltas (`exp − baseline`). Writes `results.csv` (one row per target per
   config) and `summary.json` (median speedup, mean |metric delta|, peak VRAM
   per config), and prints a concise table.
5. **Structure equivalence** (optional): if both configs produced a rank-0 CIF
   for a target, computes backbone (CA) RMSD between them using `gemmi`
   (already a Boltz dependency). Skips gracefully if `gemmi` is unimportable or
   atom counts differ. Useful for validating "numerics preserved" changes — a
   near-zero RMSD confirms the experimental change did not move atoms.

## Usage

```bash
python scripts/bench/boltz_bench.py \
    --data <yaml-or-dir> --out <results-dir> \
    --baseline-args "..." --exp-args "..." \
    [--repeat N] [--label-baseline NAME] [--label-exp NAME] [--profile]
```

### Example: default vs more diffusion samples (collapsed seed)

Compare the default sampling against 25 diffusion samples for antibody folding:

```bash
python scripts/bench/boltz_bench.py \
    --data ./antibodies/ --out ./bench_ds \
    --baseline-args "--diffusion_samples 1" \
    --exp-args     "--diffusion_samples 25 --use_potentials" \
    --label-baseline ds1 --label-exp ds25 --repeat 3
```

### Example: default vs a compiled / batched config

```bash
python scripts/bench/boltz_bench.py \
    --data ./antibodies/ --out ./bench_compiled \
    --baseline-args "" \
    --exp-args     "--no_kernels=false --sampling_steps 50" \
    --label-baseline eager --label-exp compiled --repeat 2
```

Use `--exp-args` to pass whatever flags toggle your experimental path (kernels,
step counts, recycling, batch sizing, etc.). The harness treats the args as
opaque, quoted strings appended to `boltz predict`.

## Profiling

```bash
python scripts/bench/boltz_bench.py --data ./one.yaml --out ./prof --profile
```

`--profile` runs ONE baseline fold and sets two env vars the run can read:

- `BOLTZ_BENCH_PROFILE=1`
- `BOLTZ_BENCH_TRACE=<out>/profile/chrome_trace.json`

To actually capture a `torch.profiler` trace, wrap the inference call with the
importable helper `profile_once(predict_callable, trace_path=...)` from
`boltz_bench.py`. It records the top ops by CUDA time (CPU time fallback) and a
coarse **trunk-vs-diffusion** attribution by op-name substring, and writes a
Chrome trace you can open in `chrome://tracing` / Perfetto. The helper is
importable and unit-testable without a GPU.

```python
from boltz_bench import profile_once

summary = profile_once(run_one_fold, trace_path="trace.json")
print(summary["attribution"])  # {"trunk": ..., "diffusion": ..., "other": ...}
```

## Outputs

- `results.csv` — one row per `(target, config)` with wall/VRAM/metrics.
- `summary.json` — median speedup, mean |metric delta|, peak VRAM per config,
  max CA-RMSD, target count.
- `profile/chrome_trace.json` — only when `--profile` + a profiling wrapper are
  used.
