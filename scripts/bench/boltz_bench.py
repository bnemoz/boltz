#!/usr/bin/env python
"""Standalone A/B benchmark and profiler harness for Boltz inference.

This script folds a small set of inputs under a *baseline* and an
*experimental* ``boltz predict`` configuration and reports the deltas in
wall time, peak GPU memory, and confidence metrics. It is meant to be run
on a GPU box (A100 / L40S / A6000 / B200) but all parsing / aggregation /
reporting logic is GPU-free and unit tested.

Typical use (antibody-vs-fixed-antigen high-throughput folding):

    python scripts/bench/boltz_bench.py \\
        --data ./abs/ --out ./bench_out \\
        --baseline-args "--diffusion_samples 5" \\
        --exp-args "--diffusion_samples 25" \\
        --label-baseline ds5 --label-exp ds25 --repeat 3

See ``scripts/bench/README.md`` for more examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Scalar confidence metrics extracted from each ``confidence_*.json``.
CONFIDENCE_METRICS: tuple[str, ...] = (
    "confidence_score",
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)

# Marker file a patched run may drop to report torch peak allocation (bytes).
TORCH_PEAK_MARKER = "torch_max_memory_allocated.txt"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class TargetMetrics:
    """Per-target aggregated confidence metrics (mean across models)."""

    target: str
    n_models: int
    metrics: dict[str, float]


@dataclass
class ConfigResult:
    """Result of running one Boltz configuration over all targets."""

    label: str
    args: str
    wall_total_s: Optional[float] = None
    wall_per_target_s: dict[str, float] = field(default_factory=dict)
    peak_vram_mb_smi: Optional[float] = None
    peak_vram_mb_torch: Optional[float] = None
    per_target: dict[str, TargetMetrics] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# GPU memory polling (robust to nvidia-smi being absent)
# --------------------------------------------------------------------------- #
def query_gpu_memory_mb() -> Optional[float]:
    """Return current used GPU memory in MiB via ``nvidia-smi``.

    Returns ``None`` if ``nvidia-smi`` is unavailable or fails. When multiple
    GPUs are present, the maximum across devices is returned.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


class GpuMemoryPoller:
    """Background thread polling peak GPU memory while a run is in flight."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.peak_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used = query_gpu_memory_mb()
            if used is not None:
                self.peak_mb = used if self.peak_mb is None else max(self.peak_mb, used)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMemoryPoller:
        # Probe once: if nvidia-smi is absent we never spawn a thread.
        if query_gpu_memory_mb() is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def read_torch_peak_mb(run_dir: Path) -> Optional[float]:
    """Read ``torch.cuda.max_memory_allocated`` (bytes) from a marker file.

    The marker is optional; absence simply yields ``None``.
    """
    marker = run_dir / TORCH_PEAK_MARKER
    if not marker.is_file():
        return None
    try:
        return float(marker.read_text().strip()) / (1024 * 1024)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Confidence JSON parsing & aggregation
# --------------------------------------------------------------------------- #
def _parse_confidence_filename(path: Path) -> Optional[tuple[str, int]]:
    """Extract ``(target, model_idx)`` from a ``confidence_*_model_*.json`` name.

    Returns ``None`` for files that do not match the expected pattern.
    """
    name = path.stem  # e.g. "confidence_TARGET_model_0"
    prefix = "confidence_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    marker = "_model_"
    idx = rest.rfind(marker)
    if idx == -1:
        return None
    target = rest[:idx]
    try:
        model_idx = int(rest[idx + len(marker) :])
    except ValueError:
        return None
    if not target:
        return None
    return target, model_idx


def load_confidence_file(path: Path) -> dict[str, float]:
    """Load the scalar confidence metrics from a single JSON file.

    Only the keys in :data:`CONFIDENCE_METRICS` are kept; missing keys are
    silently skipped so the harness tolerates Boltz-1 vs Boltz-2 differences.
    """
    with path.open() as f:
        data = json.load(f)
    out: dict[str, float] = {}
    for key in CONFIDENCE_METRICS:
        if key in data and isinstance(data[key], (int, float)):
            out[key] = float(data[key])
    return out


def parse_predictions_dir(predictions_dir: Path) -> dict[str, TargetMetrics]:
    """Parse all ``confidence_*.json`` under a predictions dir.

    Aggregates per target by taking the mean of each metric across the
    target's models.

    Parameters
    ----------
    predictions_dir
        A ``.../boltz_results_*/predictions`` directory (or any dir whose
        subtree contains ``confidence_*.json`` files).

    Returns
    -------
    dict
        Mapping ``target -> TargetMetrics``.
    """
    # target -> metric -> list of per-model values
    collected: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, set[int]] = {}

    for path in sorted(predictions_dir.rglob("confidence_*.json")):
        parsed = _parse_confidence_filename(path)
        if parsed is None:
            continue
        target, model_idx = parsed
        metrics = load_confidence_file(path)
        bucket = collected.setdefault(target, {})
        for key, value in metrics.items():
            bucket.setdefault(key, []).append(value)
        counts.setdefault(target, set()).add(model_idx)

    result: dict[str, TargetMetrics] = {}
    for target, bucket in collected.items():
        means = {key: statistics.fmean(vals) for key, vals in bucket.items() if vals}
        result[target] = TargetMetrics(
            target=target,
            n_models=len(counts.get(target, set())),
            metrics=means,
        )
    return result


def find_predictions_dir(out_dir: Path) -> Optional[Path]:
    """Locate the ``predictions`` dir produced by a Boltz run under ``out_dir``."""
    direct = out_dir / "predictions"
    if direct.is_dir():
        return direct
    for child in sorted(out_dir.glob("boltz_results_*")):
        cand = child / "predictions"
        if cand.is_dir():
            return cand
    # Fall back to any nested predictions dir.
    for cand in sorted(out_dir.rglob("predictions")):
        if cand.is_dir():
            return cand
    return None


# --------------------------------------------------------------------------- #
# A/B delta computation
# --------------------------------------------------------------------------- #
def compute_metric_deltas(
    baseline: dict[str, float],
    exp: dict[str, float],
) -> dict[str, float]:
    """Return ``exp - baseline`` for every metric present in both dicts."""
    return {
        key: exp[key] - baseline[key]
        for key in CONFIDENCE_METRICS
        if key in baseline and key in exp
    }


def compute_speedup(baseline_s: Optional[float], exp_s: Optional[float]) -> Optional[float]:
    """Return wall-time speedup ``baseline / exp`` (``>1`` means exp faster)."""
    if baseline_s is None or exp_s is None or exp_s <= 0:
        return None
    return baseline_s / exp_s


def compute_vram_delta(
    baseline_mb: Optional[float],
    exp_mb: Optional[float],
) -> Optional[float]:
    """Return peak-VRAM delta ``exp - baseline`` in MiB (negative = exp uses less)."""
    if baseline_mb is None or exp_mb is None:
        return None
    return exp_mb - baseline_mb


@dataclass
class TargetComparison:
    """A/B comparison for a single target."""

    target: str
    speedup: Optional[float]
    vram_delta_mb: Optional[float]
    metric_deltas: dict[str, float]
    rmsd_ca: Optional[float] = None


def build_comparison(
    baseline: ConfigResult,
    exp: ConfigResult,
) -> dict[str, TargetComparison]:
    """Build per-target A/B comparisons across the two configs."""
    targets = sorted(set(baseline.per_target) | set(exp.per_target))
    comparisons: dict[str, TargetComparison] = {}
    for target in targets:
        b_metrics = baseline.per_target.get(target)
        e_metrics = exp.per_target.get(target)
        deltas: dict[str, float] = {}
        if b_metrics is not None and e_metrics is not None:
            deltas = compute_metric_deltas(b_metrics.metrics, e_metrics.metrics)
        comparisons[target] = TargetComparison(
            target=target,
            speedup=compute_speedup(
                baseline.wall_per_target_s.get(target),
                exp.wall_per_target_s.get(target),
            ),
            vram_delta_mb=compute_vram_delta(
                baseline.peak_vram_mb_smi, exp.peak_vram_mb_smi
            ),
            metric_deltas=deltas,
        )
    return comparisons


# --------------------------------------------------------------------------- #
# Structure equivalence (optional, requires gemmi)
# --------------------------------------------------------------------------- #
def ca_rmsd(cif_a: Path, cif_b: Path) -> Optional[float]:
    """Compute backbone (CA) RMSD between two CIF structures.

    Returns ``None`` if ``gemmi`` is not importable, a file is missing, or the
    CA atom counts differ. No superposition is performed: this checks that a
    "numerics preserved" change yields the *same* coordinates, not merely a
    similar fold.
    """
    try:
        import gemmi  # noqa: PLC0415  (optional dependency)
    except ImportError:
        return None
    if not cif_a.is_file() or not cif_b.is_file():
        return None

    def _ca_coords(path: Path) -> list[tuple[float, float, float]]:
        st = gemmi.read_structure(str(path))
        coords: list[tuple[float, float, float]] = []
        if not len(st):
            return coords
        model = st[0]
        for chain in model:
            for res in chain:
                atom = res.find_atom("CA", "*")
                if atom is not None:
                    p = atom.pos
                    coords.append((p.x, p.y, p.z))
        return coords

    a = _ca_coords(cif_a)
    b = _ca_coords(cif_b)
    if not a or len(a) != len(b):
        return None
    sq = sum(
        (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        for (ax, ay, az), (bx, by, bz) in zip(a, b)
    )
    return (sq / len(a)) ** 0.5


def find_rank0_cif(predictions_dir: Path, target: str) -> Optional[Path]:
    """Locate the rank-0 (best) CIF for a target under a predictions dir."""
    cand = predictions_dir / target / f"{target}_model_0.cif"
    if cand.is_file():
        return cand
    matches = sorted(predictions_dir.rglob(f"{target}_model_0.cif"))
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# torch.profiler wrapper (importable / testable without a GPU)
# --------------------------------------------------------------------------- #
# Op-name substrings used for a coarse trunk-vs-diffusion attribution.
DIFFUSION_OP_HINTS: tuple[str, ...] = ("diffusion", "denois", "atom_attention", "sample")
TRUNK_OP_HINTS: tuple[str, ...] = ("trunk", "pairformer", "msa", "triangle", "evoformer")


def _attribute_op(name: str) -> str:
    """Classify an op name into ``"diffusion"``, ``"trunk"`` or ``"other"``."""
    low = name.lower()
    if any(h in low for h in DIFFUSION_OP_HINTS):
        return "diffusion"
    if any(h in low for h in TRUNK_OP_HINTS):
        return "trunk"
    return "other"


def profile_once(
    predict_callable: Callable[[], Any],
    trace_path: Optional[Path] = None,
    top_k: int = 25,
) -> dict[str, Any]:
    """Run ``predict_callable`` once under ``torch.profiler`` and summarize.

    Records the top ops by CUDA time (falling back to CPU time when no CUDA
    activity is present) and a coarse trunk-vs-diffusion attribution by
    op-name substring. Optionally writes a Chrome trace.

    This helper is importable and unit-testable without a GPU; if ``torch`` is
    unavailable it raises ``RuntimeError`` rather than crashing on import.

    Returns
    -------
    dict
        ``{"top_ops": [...], "attribution": {...}, "trace_path": str|None}``.
    """
    try:
        import torch  # noqa: PLC0415
        from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "torch is required for profile_once"
        raise RuntimeError(msg) from exc

    activities = [ProfilerActivity.CPU]
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        predict_callable()

    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"
    events = prof.key_averages()

    def _device_time(evt: Any) -> float:
        # cuda_time_total is in microseconds; fall back to CPU.
        return float(getattr(evt, sort_key, 0.0) or 0.0)

    ranked = sorted(events, key=_device_time, reverse=True)
    top_ops = [
        {
            "name": evt.key,
            "device_time_us": _device_time(evt),
            "cpu_time_us": float(getattr(evt, "cpu_time_total", 0.0) or 0.0),
            "count": int(getattr(evt, "count", 0) or 0),
            "bucket": _attribute_op(evt.key),
        }
        for evt in ranked[:top_k]
    ]

    attribution: dict[str, float] = {"trunk": 0.0, "diffusion": 0.0, "other": 0.0}
    for evt in events:
        attribution[_attribute_op(evt.key)] += _device_time(evt)

    return {
        "top_ops": top_ops,
        "attribution": attribution,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "used_cuda": use_cuda,
    }


# --------------------------------------------------------------------------- #
# Subprocess runner
# --------------------------------------------------------------------------- #
def run_boltz_config(
    data: Path,
    extra_args: str,
    run_out: Path,
    *,
    poll_interval_s: float = 0.25,
    env_overrides: Optional[dict[str, str]] = None,
) -> tuple[float, Optional[float], Optional[float], Path]:
    """Run ``boltz predict`` once and capture timing + peak memory.

    Returns ``(wall_s, peak_vram_mb_smi, peak_vram_mb_torch, predictions_dir)``.
    """
    import os  # noqa: PLC0415

    run_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "boltz.main",
        "predict",
        str(data),
        "--out_dir",
        str(run_out),
        *shlex.split(extra_args),
    ]
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    with GpuMemoryPoller(interval_s=poll_interval_s) as poller:
        start = time.perf_counter()
        subprocess.run(cmd, check=True, env=env)  # noqa: S603
        wall_s = time.perf_counter() - start

    predictions_dir = find_predictions_dir(run_out) or run_out
    return (
        wall_s,
        poller.peak_mb,
        read_torch_peak_mb(run_out),
        predictions_dir,
    )


def run_config(
    label: str,
    extra_args: str,
    data: Path,
    out_root: Path,
    repeat: int,
) -> ConfigResult:
    """Run one configuration ``repeat`` times and aggregate the results."""
    result = ConfigResult(label=label, args=extra_args)
    wall_totals: list[float] = []
    last_predictions: Optional[Path] = None

    for rep in range(repeat):
        run_out = out_root / f"{label}_rep{rep}"
        wall_s, vram_smi, vram_torch, predictions_dir = run_boltz_config(
            data, extra_args, run_out
        )
        wall_totals.append(wall_s)
        last_predictions = predictions_dir
        if vram_smi is not None:
            result.peak_vram_mb_smi = (
                vram_smi
                if result.peak_vram_mb_smi is None
                else max(result.peak_vram_mb_smi, vram_smi)
            )
        if vram_torch is not None:
            result.peak_vram_mb_torch = (
                vram_torch
                if result.peak_vram_mb_torch is None
                else max(result.peak_vram_mb_torch, vram_torch)
            )

    result.wall_total_s = statistics.fmean(wall_totals) if wall_totals else None
    if last_predictions is not None:
        result.per_target = parse_predictions_dir(last_predictions)
        # Even split of total wall across targets (Boltz does not emit
        # per-target timing); useful as a coarse per-target proxy.
        n = len(result.per_target) or 1
        if result.wall_total_s is not None:
            per = result.wall_total_s / n
            result.wall_per_target_s = {t: per for t in result.per_target}
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_results_csv(
    path: Path,
    baseline: ConfigResult,
    exp: ConfigResult,
) -> None:
    """Write one row per (target, config) to ``results.csv``."""
    fieldnames = [
        "config_label",
        "target",
        "n_models",
        "wall_total_s",
        "peak_vram_mb_smi",
        "peak_vram_mb_torch",
        *CONFIDENCE_METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg in (baseline, exp):
            for target, tm in sorted(cfg.per_target.items()):
                row: dict[str, Any] = {
                    "config_label": cfg.label,
                    "target": target,
                    "n_models": tm.n_models,
                    "wall_total_s": cfg.wall_total_s,
                    "peak_vram_mb_smi": cfg.peak_vram_mb_smi,
                    "peak_vram_mb_torch": cfg.peak_vram_mb_torch,
                }
                for key in CONFIDENCE_METRICS:
                    row[key] = tm.metrics.get(key)
                writer.writerow(row)


def build_summary(
    baseline: ConfigResult,
    exp: ConfigResult,
    comparisons: dict[str, TargetComparison],
) -> dict[str, Any]:
    """Build the overall ``summary.json`` payload."""
    speedups = [c.speedup for c in comparisons.values() if c.speedup is not None]
    abs_metric_deltas: dict[str, list[float]] = {}
    for comp in comparisons.values():
        for key, val in comp.metric_deltas.items():
            abs_metric_deltas.setdefault(key, []).append(abs(val))
    rmsds = [c.rmsd_ca for c in comparisons.values() if c.rmsd_ca is not None]

    return {
        "label_baseline": baseline.label,
        "label_exp": exp.label,
        "args_baseline": baseline.args,
        "args_exp": exp.args,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "mean_abs_metric_delta": {
            key: statistics.fmean(vals) for key, vals in abs_metric_deltas.items()
        },
        "peak_vram_mb_smi": {
            baseline.label: baseline.peak_vram_mb_smi,
            exp.label: exp.peak_vram_mb_smi,
        },
        "peak_vram_mb_torch": {
            baseline.label: baseline.peak_vram_mb_torch,
            exp.label: exp.peak_vram_mb_torch,
        },
        "wall_total_s": {
            baseline.label: baseline.wall_total_s,
            exp.label: exp.wall_total_s,
        },
        "max_ca_rmsd": max(rmsds) if rmsds else None,
        "n_targets": len(comparisons),
    }


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """Write the summary payload to ``summary.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)


def _fmt(value: Optional[float], spec: str = ".3f") -> str:
    """Format an optional float, rendering ``None`` as ``n/a``."""
    if value is None:
        return "n/a"
    return format(value, spec)


def print_table(
    baseline: ConfigResult,
    exp: ConfigResult,
    comparisons: dict[str, TargetComparison],
    summary: dict[str, Any],
) -> None:
    """Print a concise human-readable A/B table to stdout."""
    print()
    print(f"Boltz A/B benchmark: {baseline.label} (baseline) vs {exp.label} (exp)")
    print(f"  baseline args: {baseline.args!r}")
    print(f"  exp args:      {exp.args!r}")
    print()
    header = f"{'target':<24} {'speedup':>9} {'dVRAM(MB)':>11} {'dconf':>9} {'dCA-RMSD':>9}"
    print(header)
    print("-" * len(header))
    for target in sorted(comparisons):
        comp = comparisons[target]
        dconf = comp.metric_deltas.get("confidence_score")
        print(
            f"{target:<24} "
            f"{_fmt(comp.speedup, '.2f'):>9} "
            f"{_fmt(comp.vram_delta_mb, '.0f'):>11} "
            f"{_fmt(dconf, '+.3f'):>9} "
            f"{_fmt(comp.rmsd_ca, '.3f'):>9}"
        )
    print("-" * len(header))
    print(f"median speedup:        {_fmt(summary['median_speedup'], '.2f')}x")
    print(
        f"peak VRAM (smi):       "
        f"{baseline.label}={_fmt(baseline.peak_vram_mb_smi, '.0f')}MB  "
        f"{exp.label}={_fmt(exp.peak_vram_mb_smi, '.0f')}MB"
    )
    mad = summary["mean_abs_metric_delta"].get("confidence_score")
    print(f"mean |dconfidence|:    {_fmt(mad, '.4f')}")
    print(f"max CA-RMSD:           {_fmt(summary['max_ca_rmsd'], '.3f')} A")
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI parser."""
    parser = argparse.ArgumentParser(
        description="A/B benchmark + profiler harness for Boltz inference.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="A Boltz input YAML/FASTA file or a directory of inputs.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Results directory (created if missing).",
    )
    parser.add_argument(
        "--baseline-args",
        default="",
        help='Quoted extra "boltz predict" flags for the baseline config.',
    )
    parser.add_argument(
        "--exp-args",
        default="",
        help='Quoted extra "boltz predict" flags for the experimental config.',
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats per config (wall time is averaged). Default 1.",
    )
    parser.add_argument(
        "--label-baseline",
        default="baseline",
        help="Short label for the baseline config.",
    )
    parser.add_argument(
        "--label-exp",
        default="exp",
        help="Short label for the experimental config.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run ONE baseline fold under torch.profiler and write a chrome "
            "trace + op summary. Requires a torch-enabled environment."
        ),
    )
    return parser


def _profile_subprocess(args: argparse.Namespace) -> None:
    """Run a single profiled baseline fold via subprocess + env knob.

    The harness sets ``BOLTZ_BENCH_PROFILE`` and a trace path env var; a
    cooperating run (or a wrapper) may read these to wrap inference in
    :func:`profile_once`. This keeps the dependency optional and the default
    path GPU-free.
    """
    trace_path = args.out / "profile" / "chrome_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile_out = args.out / "profile" / "profile_run"
    print(f"[profile] running one baseline fold; trace -> {trace_path}")
    print(
        "[profile] note: set BOLTZ_BENCH_PROFILE / BOLTZ_BENCH_TRACE in a "
        "wrapper to capture a torch.profiler trace. See README."
    )
    run_boltz_config(
        args.data,
        args.baseline_args,
        profile_out,
        env_overrides={
            "BOLTZ_BENCH_PROFILE": "1",
            "BOLTZ_BENCH_TRACE": str(trace_path),
        },
    )


def attach_rmsd(
    baseline: ConfigResult,
    exp: ConfigResult,
    comparisons: dict[str, TargetComparison],
    baseline_predictions: Optional[Path],
    exp_predictions: Optional[Path],
) -> None:
    """Populate CA-RMSD on comparisons when both configs produced a CIF."""
    if baseline_predictions is None or exp_predictions is None:
        return
    for target, comp in comparisons.items():
        cif_a = find_rank0_cif(baseline_predictions, target)
        cif_b = find_rank0_cif(exp_predictions, target)
        if cif_a is not None and cif_b is not None:
            comp.rmsd_ca = ca_rmsd(cif_a, cif_b)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.profile:
        _profile_subprocess(args)
        return 0

    baseline = run_config(
        args.label_baseline, args.baseline_args, args.data, args.out, args.repeat
    )
    exp = run_config(
        args.label_exp, args.exp_args, args.data, args.out, args.repeat
    )

    comparisons = build_comparison(baseline, exp)

    baseline_preds = find_predictions_dir(args.out / f"{baseline.label}_rep0")
    exp_preds = find_predictions_dir(args.out / f"{exp.label}_rep0")
    attach_rmsd(baseline, exp, comparisons, baseline_preds, exp_preds)

    summary = build_summary(baseline, exp, comparisons)
    write_results_csv(args.out / "results.csv", baseline, exp)
    write_summary_json(args.out / "summary.json", summary)
    print_table(baseline, exp, comparisons, summary)
    print(f"Wrote {args.out / 'results.csv'} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
