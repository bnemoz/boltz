# GPU validation A/B matrix

Turnkey commands to validate each optimization the moment a GPU frees. All use
`scripts/bench/boltz_bench.py` (this branch). Pick a held-out set `$DATA` of ~10–20
representative antibody+antigen YAMLs (the real antigen, varied Fv) and a scratch `$OUT`.

**Acceptance:** wins must hold quality. For `IDENTICAL`/`FP-TOL` items, per-candidate
`complex_iplddt`/`iptm`/`ptm` deltas ≈ 0 (FP-TOL: within tolerance) and CA-RMSD ≈ 0 between
rank-0 models; only the wall/VRAM should move. Record wall, peak VRAM, workers/GPU.

Two A/B styles:
- **flag A/B (same install):** `--baseline-args`/`--exp-args` on one checkout — for knobs.
- **install A/B (two checkouts):** run the harness twice with identical args, once on `main`
  and once on the branch, then diff the two `results.csv` — for code that isn't flag-gated
  (e.g. SDPA default-on).

## 1. compile-score (#3) — flag A/B on the `opt/compile-score` checkout
`T`/`A` = token/atom counts ≥ your largest complex (so it compiles once).
```
boltz_bench.py --data $DATA --out $OUT/compile \
  --label-baseline noncompiled --baseline-args "--pad_to_tokens T --pad_to_atoms A" \
  --label-exp compiled        --exp-args      "--compile --pad_to_tokens T --pad_to_atoms A" \
  --repeat 2
```
Expect: diffusion wall ↓ ~10–20% after the first (warmup) fold; metrics/RMSD ≈ 0. The first
fold pays compile cost — use `--repeat 2` and read the 2nd.

## 2. sdpa-attention (#4) — install A/B (main vs branch), identical args
```
# on main checkout:            boltz_bench.py --data $DATA --out $OUT/sdpa_base --label-baseline main
# on opt/sdpa-attention:       boltz_bench.py --data $DATA --out $OUT/sdpa_exp  --label-exp sdpa
# then compare the two results.csv (wall, peak VRAM, metric deltas, CA-RMSD)
```
Expect: peak VRAM ↓ (no N×N score buffer) + some wall ↓; fp32 metrics ≈ identical. Also A/B
bf16 attention separately and confirm ranking stability before adopting bf16.

## 3. batch-inference (#5) — flag A/B on the `opt/batch-inference` checkout
`$DATA` must be a *directory* of many YAMLs.
```
boltz_bench.py --data $DATA --out $OUT/batch \
  --label-baseline b1 --baseline-args "--batch_size 1" \
  --label-exp b4      --exp-args      "--batch_size 4" \
  --repeat 1
```
Expect: higher antibodies/GPU-hour on A100-80G/B200 (limited on 48 GiB cards); per-record
outputs identical to B=1. Sanity-check a couple of CIFs match the B=1 run (mask no-leak).

## 4. seed-collapse + max_parallel_samples (mab-design config) — pipeline-level A/B
Not a Boltz flag — compare two `config/pipeline.yaml` folding blocks via the real fold stage:
- baseline: `seeds:[1,42,89,1337,2026]`, `diffusion_samples:5`, `max_parallel_samples:5`
- exp:      `seeds:[1]`, `diffusion_samples:25`, `max_parallel_samples:2`
Fold the same ~10 candidates both ways; compare per-candidate confidence aggregates +
ssRMSD spread (expect statistically identical) and total wall + workers/GPU (expect ↓).
Re-measure `fold_peak_gib` under the exp config.

## 5. profile one fold (confirm the diffusion-bound split per arch)
```
boltz_bench.py --data <one.yaml> --out $OUT/profile --profile
```
Reads the chrome trace + coarse trunk-vs-diffusion attribution; re-confirm on L40S/A6000 vs
A100 vs B200 before investing further (e.g. whether kernels help on B200).

## Suggested order
profile (5) → seed-collapse (4, biggest, config-only) → compile (1) → batch (3) → sdpa (2).
Then, if validated, build an `opt/all` branch combining the code wins for the deployed stack.
