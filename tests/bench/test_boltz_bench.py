"""Unit tests for the GPU-free logic of ``scripts/bench/boltz_bench.py``.

These tests exercise confidence-JSON parsing + per-target aggregation, A/B
delta computation, CSV/summary writing, and the nvidia-smi-absent path. No
subprocess and no GPU are used.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Import the harness module directly from scripts/bench (not a package).
_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "boltz_bench.py"
_spec = importlib.util.spec_from_file_location("boltz_bench", _BENCH)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
sys.modules["boltz_bench"] = bench
_spec.loader.exec_module(bench)


def _write_confidence(
    pred_dir: Path,
    target: str,
    model_idx: int,
    values: dict[str, float],
) -> None:
    """Write a synthetic ``confidence_<target>_model_<idx>.json`` file."""
    target_dir = pred_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"confidence_{target}_model_{model_idx}.json"
    # Include extra non-scalar keys to confirm they are ignored.
    payload = dict(values)
    payload["chains_ptm"] = {"0": 0.9}
    path.write_text(json.dumps(payload))


# --------------------------------------------------------------------------- #
# (a) confidence parsing + per-target aggregation
# --------------------------------------------------------------------------- #
def test_parse_and_aggregate(tmp_path: Path) -> None:
    pred = tmp_path / "predictions"
    # Target A: two models -> means.
    _write_confidence(pred, "abA", 0, {"confidence_score": 0.8, "ptm": 0.7, "iptm": 0.6})
    _write_confidence(pred, "abA", 1, {"confidence_score": 0.6, "ptm": 0.5, "iptm": 0.4})
    # Target B: single model.
    _write_confidence(pred, "abB", 0, {"confidence_score": 0.9, "complex_plddt": 0.85})

    result = bench.parse_predictions_dir(pred)
    assert set(result) == {"abA", "abB"}

    a = result["abA"]
    assert a.n_models == 2
    assert a.metrics["confidence_score"] == pytest.approx(0.7)
    assert a.metrics["ptm"] == pytest.approx(0.6)
    assert a.metrics["iptm"] == pytest.approx(0.5)

    b = result["abB"]
    assert b.n_models == 1
    assert b.metrics["confidence_score"] == pytest.approx(0.9)
    assert b.metrics["complex_plddt"] == pytest.approx(0.85)
    # Metric absent from the file is not invented.
    assert "iptm" not in b.metrics


def test_parse_filename_edge_cases() -> None:
    # Target names may themselves contain "_model_"-like substrings; we split
    # on the *last* occurrence.
    parsed = bench._parse_confidence_filename(Path("confidence_my_model_x_model_3.json"))
    assert parsed == ("my_model_x", 3)
    assert bench._parse_confidence_filename(Path("not_confidence.json")) is None
    assert bench._parse_confidence_filename(Path("confidence_foo.json")) is None


def test_find_predictions_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    nested = out / "boltz_results_inputs" / "predictions"
    nested.mkdir(parents=True)
    assert bench.find_predictions_dir(out) == nested


# --------------------------------------------------------------------------- #
# (b) A/B delta computation
# --------------------------------------------------------------------------- #
def test_compute_metric_deltas() -> None:
    baseline = {"confidence_score": 0.80, "ptm": 0.70, "iptm": 0.60}
    exp = {"confidence_score": 0.83, "ptm": 0.68, "iptm": 0.60}
    deltas = bench.compute_metric_deltas(baseline, exp)
    assert deltas["confidence_score"] == pytest.approx(0.03)
    assert deltas["ptm"] == pytest.approx(-0.02)
    assert deltas["iptm"] == pytest.approx(0.0)


def test_compute_speedup_and_vram() -> None:
    assert bench.compute_speedup(10.0, 5.0) == pytest.approx(2.0)
    assert bench.compute_speedup(None, 5.0) is None
    assert bench.compute_speedup(10.0, 0.0) is None
    assert bench.compute_vram_delta(1000.0, 800.0) == pytest.approx(-200.0)
    assert bench.compute_vram_delta(None, 800.0) is None


def test_build_comparison() -> None:
    baseline = bench.ConfigResult(label="base", args="")
    exp = bench.ConfigResult(label="exp", args="--diffusion_samples 25")
    baseline.per_target = {
        "abA": bench.TargetMetrics("abA", 1, {"confidence_score": 0.80, "ptm": 0.70}),
    }
    exp.per_target = {
        "abA": bench.TargetMetrics("abA", 1, {"confidence_score": 0.85, "ptm": 0.72}),
    }
    baseline.wall_per_target_s = {"abA": 20.0}
    exp.wall_per_target_s = {"abA": 10.0}
    baseline.peak_vram_mb_smi = 4000.0
    exp.peak_vram_mb_smi = 5000.0

    comps = bench.build_comparison(baseline, exp)
    comp = comps["abA"]
    assert comp.speedup == pytest.approx(2.0)
    assert comp.vram_delta_mb == pytest.approx(1000.0)
    assert comp.metric_deltas["confidence_score"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# (c) CSV / summary writing
# --------------------------------------------------------------------------- #
def _two_config_fixture() -> tuple[object, object, dict]:
    baseline = bench.ConfigResult(label="base", args="", wall_total_s=20.0)
    exp = bench.ConfigResult(label="exp", args="--x", wall_total_s=10.0)
    baseline.peak_vram_mb_smi = 4000.0
    exp.peak_vram_mb_smi = 4200.0
    baseline.per_target = {
        "abA": bench.TargetMetrics("abA", 2, {"confidence_score": 0.80, "ptm": 0.70}),
        "abB": bench.TargetMetrics("abB", 1, {"confidence_score": 0.60}),
    }
    exp.per_target = {
        "abA": bench.TargetMetrics("abA", 2, {"confidence_score": 0.85, "ptm": 0.71}),
        "abB": bench.TargetMetrics("abB", 1, {"confidence_score": 0.61}),
    }
    baseline.wall_per_target_s = {"abA": 10.0, "abB": 10.0}
    exp.wall_per_target_s = {"abA": 5.0, "abB": 5.0}
    comps = bench.build_comparison(baseline, exp)
    return baseline, exp, comps


def test_write_results_csv(tmp_path: Path) -> None:
    baseline, exp, _ = _two_config_fixture()
    csv_path = tmp_path / "results.csv"
    bench.write_results_csv(csv_path, baseline, exp)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    # 2 targets x 2 configs = 4 rows.
    assert len(rows) == 4
    labels = {(r["config_label"], r["target"]) for r in rows}
    assert ("base", "abA") in labels
    assert ("exp", "abB") in labels
    a_exp = next(r for r in rows if r["config_label"] == "exp" and r["target"] == "abA")
    assert float(a_exp["confidence_score"]) == pytest.approx(0.85)
    assert float(a_exp["peak_vram_mb_smi"]) == pytest.approx(4200.0)


def test_build_and_write_summary(tmp_path: Path) -> None:
    baseline, exp, comps = _two_config_fixture()
    summary = bench.build_summary(baseline, exp, comps)
    # Both targets have 2x speedup -> median 2.0.
    assert summary["median_speedup"] == pytest.approx(2.0)
    assert summary["n_targets"] == 2
    assert summary["mean_abs_metric_delta"]["confidence_score"] == pytest.approx(0.03)
    assert summary["peak_vram_mb_smi"]["base"] == pytest.approx(4000.0)

    summary_path = tmp_path / "summary.json"
    bench.write_summary_json(summary_path, summary)
    loaded = json.loads(summary_path.read_text())
    assert loaded["label_exp"] == "exp"
    assert loaded["median_speedup"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# (d) nvidia-smi-absent path
# --------------------------------------------------------------------------- #
def test_query_gpu_memory_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(bench.subprocess, "check_output", _raise)
    assert bench.query_gpu_memory_mb() is None


def test_poller_no_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "query_gpu_memory_mb", lambda: None)
    with bench.GpuMemoryPoller(interval_s=0.01) as poller:
        pass
    assert poller.peak_mb is None


def test_poller_with_values(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([1000.0, 3000.0, 2000.0])

    def _next() -> float:
        try:
            return next(values)
        except StopIteration:
            return 2000.0

    monkeypatch.setattr(bench, "query_gpu_memory_mb", _next)
    with bench.GpuMemoryPoller(interval_s=0.005) as poller:
        import time

        time.sleep(0.05)
    assert poller.peak_mb is not None
    assert poller.peak_mb >= 2000.0


def test_read_torch_peak_marker(tmp_path: Path) -> None:
    assert bench.read_torch_peak_mb(tmp_path) is None
    (tmp_path / bench.TORCH_PEAK_MARKER).write_text(str(2 * 1024 * 1024))
    assert bench.read_torch_peak_mb(tmp_path) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# profiler attribution (no GPU / no torch needed)
# --------------------------------------------------------------------------- #
def test_attribute_op() -> None:
    assert bench._attribute_op("diffusion_step") == "diffusion"
    assert bench._attribute_op("PairformerLayer") == "trunk"
    assert bench._attribute_op("aten::add") == "other"


def test_ca_rmsd_missing_files(tmp_path: Path) -> None:
    # Missing files -> None regardless of gemmi availability.
    assert bench.ca_rmsd(tmp_path / "a.cif", tmp_path / "b.cif") is None
