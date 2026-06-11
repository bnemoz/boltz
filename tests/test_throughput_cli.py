"""Tests for the throughput-cli knobs: ``--compile`` and ``--batch_size``.

These tests are GPU-free and data-free: they never load a model checkpoint
nor run inference. They only exercise the CLI option surface and the
DataModule plumbing of the ``batch_size`` parameter.
"""

from pathlib import Path

from click.testing import CliRunner

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.main import cli


def test_predict_help_exposes_compile_and_batch_size() -> None:
    """`boltz predict --help` advertises the new options with true defaults."""
    result = CliRunner().invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0, result.output

    # Compile is an on/off flag, defaulting to OFF.
    assert "--compile / --no-compile" in result.output
    assert "Default is False." in result.output

    # Batch size is an int defaulting to 1 (preserves current behavior).
    assert "--batch_size INTEGER" in result.output
    assert "Default is 1" in result.output


def test_boltz2_datamodule_default_batch_size() -> None:
    """Boltz2 DataModule defaults batch_size to 1 (current behavior)."""
    dm = Boltz2InferenceDataModule(
        manifest=None,
        target_dir=Path("/tmp/targets"),
        msa_dir=Path("/tmp/msa"),
        mol_dir=Path("/tmp/mol"),
        num_workers=0,
    )
    assert dm.batch_size == 1


def test_boltz2_datamodule_stores_batch_size() -> None:
    """Boltz2 DataModule stores an explicit batch_size."""
    dm = Boltz2InferenceDataModule(
        manifest=None,
        target_dir=Path("/tmp/targets"),
        msa_dir=Path("/tmp/msa"),
        mol_dir=Path("/tmp/mol"),
        num_workers=0,
        batch_size=4,
    )
    assert dm.batch_size == 4


def test_boltz1_datamodule_default_and_stored_batch_size() -> None:
    """Boltz1 DataModule defaults batch_size to 1 and stores overrides."""
    dm_default = BoltzInferenceDataModule(
        manifest=None,
        target_dir=Path("/tmp/targets"),
        msa_dir=Path("/tmp/msa"),
        num_workers=0,
    )
    assert dm_default.batch_size == 1

    dm_override = BoltzInferenceDataModule(
        manifest=None,
        target_dir=Path("/tmp/targets"),
        msa_dir=Path("/tmp/msa"),
        num_workers=0,
        batch_size=8,
    )
    assert dm_override.batch_size == 8


def test_predict_dataloader_uses_stored_batch_size(monkeypatch) -> None:
    """`predict_dataloader` builds the DataLoader with the stored batch_size.

    We monkeypatch the dataset constructor and the DataLoader so the test
    stays data-free (no rdkit molecule loading, no disk access).
    """
    import boltz.data.module.inferencev2 as mod

    captured = {}

    class _DummyDataLoader:
        def __init__(self, dataset, *, batch_size, **kwargs):
            captured["batch_size"] = batch_size

    monkeypatch.setattr(mod, "PredictionDataset", lambda **kwargs: object())
    monkeypatch.setattr(mod, "DataLoader", _DummyDataLoader)

    dm = Boltz2InferenceDataModule(
        manifest=None,
        target_dir=Path("/tmp/targets"),
        msa_dir=Path("/tmp/msa"),
        mol_dir=Path("/tmp/mol"),
        num_workers=0,
        batch_size=3,
    )
    dm.predict_dataloader()
    assert captured["batch_size"] == 3
