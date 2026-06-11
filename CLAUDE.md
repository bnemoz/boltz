# CLAUDE.md — Boltz inference notes (high-throughput antibody screening)

Operational knowledge for running/optimizing Boltz-2 inference at scale (e.g. folding many
antibody candidates against one fixed antigen). Hard-won notes that belong with the code, plus
the optimization branches in this fork.

## Performance profile (where the time goes)

A ~700-token complex fold is **diffusion-bound**: ~75% is the reverse-diffusion sampling loop
(`sampling_steps`, default 200), ~20% the 48-block Pairformer trunk (× `recycling_steps+1`
passes), ~8% the separate confidence Pairformer, ~4% input/MSA encoding. **Optimize the diffusion
side first.** The model is loaded once and reused across all targets in a `boltz predict` run.

## Optimization branches in this fork (see PRs)

| Branch | Lever | Numerics |
|---|---|---|
| `opt/benchmark-harness` | A/B + profiler harness (`scripts/bench/boltz_bench.py`) | tooling |
| `opt/throughput-cli` | `--compile`, `--batch_size` flags | identical at defaults |
| `opt/compile-score` | `compile_structure` + `--pad_to_tokens/--pad_to_atoms` (compile once) | identical |
| `opt/sdpa-attention` | `AttentionPairBias` → `scaled_dot_product_attention` | fp32 exact / bf16 within tol |
| `opt/batch-inference` | `batch_size > 1` on the predict path (no affinity) | identical for B==1 |

Config-only wins (no code) for a multi-seed × multi-sample screen:
- **Collapse seeds into `diffusion_samples`.** At inference the trunk is deterministic (no eval
  dropout, `subsample_msa` off, no cropping), so structural diversity comes *only* from diffusion
  noise. `seeds=[1]` + `diffusion_samples=N` gives the same ensemble as `N` seeds × 1 sample but
  pays startup + trunk + confidence once per candidate, not N×.
- **Decouple `max_parallel_samples` from `diffusion_samples`.** It's a pure VRAM/chunking knob —
  never changes results. Lower it to fit more concurrent folds per GPU.
- **Keep the model resident** across many targets (one `boltz predict <dir>` per GPU, or a worker
  loop) instead of one process per (candidate, seed) — startup is ~15–40 s each.

## Footguns / gotchas

- **Cold CCD cache + parallel launches corrupt the cache** ("CCD component ALA not found!"). Pre-warm
  the shared `$BOLTZ_CACHE` **serially** (one tiny fold) before fanning out parallel workers.
- **`--subsample_msa` effectively defaults OFF.** It's a click `is_flag`, so absence ⇒ `False`,
  despite the `subsample_msa: bool = True` in the function signature and the "Default is True" help
  text. Pass `--subsample_msa` to actually subsample (stochastic; ~98% cheaper MSA blocks on a deep
  antigen MSA).
- **`--write_full_pae` help says "Default is True" but it's a flag ⇒ actually `False`.** Same for
  `--write_full_pde`. So PAE/PDE npz are NOT written unless you pass the flags (good for I/O).
- **CUDA triangle kernels (`--no_kernels` off by default) gave no end-to-end speedup on L40S/A6000**
  because the wall is diffusion-bound and kernels only accelerate the trunk/confidence. They matter
  more the more you pay the trunk (e.g. without seed-collapse). Worth retesting on B200 (SM 10.0).
- **I/O:** write Boltz's full per-prediction output to **local scratch** and copy only the CIF +
  `confidence_*.json` back to slow shared storage; the run stays GPU/scratch-bound, not FS-bound.

## Precision

Boltz-2 already runs **bf16-mixed** (`main.py`); Boltz-1 runs fp32. There is **no fp64 anywhere** in
the inference path — the only forced-fp32 spots (Kabsch/SVD alignment, distogram binning) are tiny
and necessary. Don't go below bf16 (fp16 risks NaNs; the model is trained for bf16's range).

## Dead end: reusing the antigen embedding across antibodies

Tempting (the antigen is fixed) but **not worth it (<0.5% end-to-end)**. The atom encoder keeps the
antigen's `s_inputs` antigen-only and the antigen×antigen `z_init` block is reusable, but the **first
Pairformer block entangles it** with the antibody (triangle multiply sums over all tokens; triangle
attention is global). Only the ~12 ms of pre-Pairformer inputs are reusable, and the trunk is a
minority of a diffusion-bound fold. The MSA *fetch* is the only antigen artifact worth caching (do
that once, off-line).
