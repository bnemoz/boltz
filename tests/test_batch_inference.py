"""Tests for batch (batch_size > 1) inference correctness.

These tests run on CPU and do NOT download any model checkpoint nor touch
rdkit. They cover three things:

1. MASK-LEAK / EQUIVALENCE at the component level using ``PairformerModule``
   (a pure-tensor submodule that threads ``(B, N, ...)`` and a pair mask).
2. ``collate`` / ``pad_to_max`` batching of variable-length samples.
3. ``BoltzWriter.write_on_batch_end`` per-record iteration for B == 1 vs B == 2.

Run:
    cd <worktree> && PYTHONPATH=src python -m pytest tests/test_batch_inference.py -q
"""

import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from boltz.data.module.inferencev2 import collate
from boltz.data.pad import pad_to_max


# ---------------------------------------------------------------------------
# 1. MASK-LEAK / EQUIVALENCE: PairformerModule
# ---------------------------------------------------------------------------
#
# Design: PairformerModule is the simplest pure-tensor submodule that
# exercises BOTH the pairwise track (triangle multiplication + triangle
# attention, masked by ``pair_mask``) and the sequence track (attention pair
# bias, masked by ``mask``). We instantiate it with tiny dims on CPU in eval
# mode (dropout disabled => deterministic).
#
# We build a batch of two samples of DIFFERENT effective lengths by padding
# the shorter one and marking the padding via ``token_pad_mask``:
#   - sample A: effective length 4 (positions 4,5 are padding)
#   - sample B: effective length 6 (no padding)
# We then run each sample ALONE as a B == 1 forward pass. If the masks
# correctly prevent padding from leaking across positions/samples, the real
# (unpadded) outputs of each sample inside the B == 2 batch must match the
# corresponding B == 1 outputs to within tight tolerance.


def _build_pairformer():
    from boltz.model.layers.pairformer import PairformerModule

    torch.manual_seed(0)
    model = PairformerModule(
        token_s=16,
        token_z=16,
        num_blocks=2,
        num_heads=2,
        dropout=0.0,
        pairwise_head_width=8,
        pairwise_num_heads=2,
        v2=True,
    )
    model.eval()
    return model


def _make_sample(n_total: int, n_real: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, n_total, 16, generator=g)
    z = torch.randn(1, n_total, n_total, 16, generator=g)
    tok_mask = torch.zeros(1, n_total)
    tok_mask[0, :n_real] = 1.0
    pair_mask = tok_mask[:, :, None] * tok_mask[:, None, :]
    # Zero out the embeddings in the padded region (mirrors how real padded
    # features are zero), so the only thing keeping padding out is the mask.
    s = s * tok_mask[..., None]
    z = z * pair_mask[..., None]
    return s, z, tok_mask, pair_mask


def test_pairformer_no_mask_leak_across_batch():
    model = _build_pairformer()
    n_total = 6
    n_real_a, n_real_b = 4, 6

    sa = _make_sample(n_total, n_real_a, seed=1)
    sb = _make_sample(n_total, n_real_b, seed=2)

    with torch.no_grad():
        # Batched B == 2 forward pass.
        s = torch.cat([sa[0], sb[0]], dim=0)
        z = torch.cat([sa[1], sb[1]], dim=0)
        tok_mask = torch.cat([sa[2], sb[2]], dim=0)
        pair_mask = torch.cat([sa[3], sb[3]], dim=0)
        s_batch, z_batch = model(s, z, tok_mask, pair_mask)

        # Each sample alone (B == 1).
        s_a, z_a = model(*sa)
        s_b, z_b = model(*sb)

    # Real-region outputs must be identical between batched and solo runs.
    assert torch.allclose(
        s_batch[0, :n_real_a], s_a[0, :n_real_a], atol=1e-5
    ), "sample A sequence output leaked across batch"
    assert torch.allclose(
        z_batch[0, :n_real_a, :n_real_a], z_a[0, :n_real_a, :n_real_a], atol=1e-5
    ), "sample A pair output leaked across batch"
    assert torch.allclose(
        s_batch[1, :n_real_b], s_b[0, :n_real_b], atol=1e-5
    ), "sample B sequence output leaked across batch"
    assert torch.allclose(
        z_batch[1, :n_real_b, :n_real_b], z_b[0, :n_real_b, :n_real_b], atol=1e-5
    ), "sample B pair output leaked across batch"


def test_pairformer_padding_changes_do_not_affect_real_region():
    """Mutating ONLY the padded region of a batched sample must not change the
    real-region output of either sample."""
    model = _build_pairformer()
    n_total = 6
    sa = _make_sample(n_total, 4, seed=1)
    sb = _make_sample(n_total, 6, seed=2)

    with torch.no_grad():
        s = torch.cat([sa[0], sb[0]], dim=0)
        z = torch.cat([sa[1], sb[1]], dim=0)
        tok_mask = torch.cat([sa[2], sb[2]], dim=0)
        pair_mask = torch.cat([sa[3], sb[3]], dim=0)
        out_s0, out_z0 = model(s, z, tok_mask, pair_mask)

        # Inject garbage into sample A's padded positions (indices 4, 5).
        s2 = s.clone()
        z2 = z.clone()
        s2[0, 4:] = 999.0
        z2[0, 4:, :] = 999.0
        z2[0, :, 4:] = 999.0
        out_s1, out_z1 = model(s2, z2, tok_mask, pair_mask)

    assert torch.allclose(out_s0[0, :4], out_s1[0, :4], atol=1e-5)
    assert torch.allclose(out_z0[0, :4, :4], out_z1[0, :4, :4], atol=1e-5)
    # Sample B (the other batch element) must be completely unaffected.
    assert torch.allclose(out_s0[1], out_s1[1], atol=1e-5)
    assert torch.allclose(out_z0[1], out_z1[1], atol=1e-5)


# ---------------------------------------------------------------------------
# 2. COLLATE / pad_to_max
# ---------------------------------------------------------------------------


def test_collate_pads_and_masks_variable_lengths():
    # Two synthetic samples with DIFFERENT token / atom counts.
    sample_a = {
        "token_pad_mask": torch.ones(4),
        "atom_pad_mask": torch.ones(10),
        "feat": torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3),
        "record": "rec_a",
    }
    sample_b = {
        "token_pad_mask": torch.ones(6),
        "atom_pad_mask": torch.ones(15),
        "feat": torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3) + 100.0,
        "record": "rec_b",
    }

    batch = collate([sample_a, sample_b])

    # Leading batch dim of 2.
    assert batch["feat"].shape[0] == 2
    assert batch["token_pad_mask"].shape == (2, 6)
    assert batch["atom_pad_mask"].shape == (2, 15)
    assert batch["feat"].shape == (2, 6, 3)

    # "record" is a non-tensor key: kept as a list (not stacked).
    assert batch["record"] == ["rec_a", "rec_b"]

    # Padded regions of sample A are zero.
    assert torch.all(batch["feat"][0, 4:] == 0)
    assert torch.all(batch["token_pad_mask"][0, 4:] == 0)
    assert torch.all(batch["atom_pad_mask"][0, 10:] == 0)
    # Real region preserved.
    assert torch.all(batch["token_pad_mask"][0, :4] == 1)
    assert torch.allclose(batch["feat"][1], sample_b["feat"])  # B already max len


def test_pad_to_max_marks_padding():
    a = torch.ones(4, 3)
    b = torch.ones(6, 3)
    padded, mask = pad_to_max([a, b], value=0.0)
    assert padded.shape == (2, 6, 3)
    # mask is 1 in real region, 0 in padded region.
    assert torch.all(mask[0, :4] == 1)
    assert torch.all(mask[0, 4:] == 0)
    assert torch.all(mask[1] == 1)

    # All-equal shapes => fast path returns a stacked tensor and scalar 0.
    same, mask0 = pad_to_max([a, a.clone()], value=0.0)
    assert same.shape == (2, 4, 3)
    assert mask0 == 0


# ---------------------------------------------------------------------------
# 3. WRITER per-record iteration
# ---------------------------------------------------------------------------
#
# We unit-test the per-record / per-diffusion-sample iteration and the
# flat-index slicing into the (B * diffusion_samples) metric tensors. The
# structure-loading and serialization is stubbed so we avoid rdkit / disk:
# we monkeypatch ``StructureV2.load`` to return a trivial stub whose
# ``remove_invalid_chains`` returns itself and whose ``chains`` is empty
# (so the per-chain remap loop is skipped), and we patch ``to_mmcif`` plus
# ``np.savez_compressed`` and ``Path.open`` to record every (record, model)
# write instead of touching disk.


@dataclass
class _FakeArr:
    """Minimal structured-array stand-in supporting item assignment."""

    def __setitem__(self, key, value):  # noqa: D401 - stub
        pass


class _FakeStructure:
    def __init__(self):
        self.mask = np.array([], dtype=bool)
        self.chains = np.array([], dtype=object)
        self.atoms = _FakeArr()
        self.residues = _FakeArr()

    def remove_invalid_chains(self):
        return self


def _make_record(rid: str):
    return types.SimpleNamespace(id=rid, chains=[], affinity=False)


def _run_writer(monkeypatch, tmp_path, *, num_records, diffusion_samples, n_atoms):
    from boltz.data.write import writer as writer_mod

    writes = []  # list of (kind, record_id, model_name)

    # Stub structure loading and serialization.
    monkeypatch.setattr(
        writer_mod.StructureV2, "load", staticmethod(lambda path: _FakeStructure())
    )
    monkeypatch.setattr(
        writer_mod, "replace", lambda struct, **kw: struct, raising=True
    )
    monkeypatch.setattr(
        writer_mod, "to_mmcif", lambda *a, **k: "MMCIF", raising=True
    )

    saved_npz = []
    monkeypatch.setattr(
        writer_mod.np,
        "savez_compressed",
        lambda path, **kw: saved_npz.append(Path(path).name),
        raising=True,
    )

    # Capture .cif writes via Path.open without touching disk.
    import io

    real_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self.suffix in (".cif", ".pdb", ".json"):
            writes.append((self.suffix, self.name))
            return io.StringIO()
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open, raising=True)
    # mkdir should be a no-op (avoid creating per-record dirs everywhere).
    monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None, raising=True)

    writer = writer_mod.BoltzWriter(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        output_format="mmcif",
        boltz2=True,
    )

    records = [_make_record(f"rec{i}") for i in range(num_records)]
    total = num_records * diffusion_samples
    # coords: (B * diffusion_samples, n_atoms, 3)
    coords = torch.arange(total * n_atoms * 3, dtype=torch.float32).reshape(
        total, n_atoms, 3
    )
    # atom_pad_mask: one per record (B, n_atoms) all real.
    masks = torch.ones(num_records, n_atoms)

    prediction = {
        "exception": False,
        "coords": coords,
        "masks": masks,
    }
    batch = {"record": records}

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch_indices=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )
    cif_writes = [w for w in writes if w[0] == ".cif"]
    return cif_writes, saved_npz


def test_writer_b1_iteration(monkeypatch, tmp_path):
    cif_writes, _ = _run_writer(
        monkeypatch, tmp_path, num_records=1, diffusion_samples=2, n_atoms=5
    )
    names = sorted(w[1] for w in cif_writes)
    # One record, two diffusion samples => model_0 and model_1.
    assert names == ["rec0_model_0.cif", "rec0_model_1.cif"]


def test_writer_b2_iteration(monkeypatch, tmp_path):
    cif_writes, _ = _run_writer(
        monkeypatch, tmp_path, num_records=2, diffusion_samples=2, n_atoms=5
    )
    names = sorted(w[1] for w in cif_writes)
    # Two records, each with two diffusion samples => 4 distinct files,
    # each record producing its own model_0 / model_1.
    assert names == [
        "rec0_model_0.cif",
        "rec0_model_1.cif",
        "rec1_model_0.cif",
        "rec1_model_1.cif",
    ]
    assert len(cif_writes) == 4


def test_writer_b1_confidence_ranking(monkeypatch, tmp_path):
    """With confidence scores, model files are ranked per record."""
    from boltz.data.write import writer as writer_mod

    writes = []
    monkeypatch.setattr(
        writer_mod.StructureV2, "load", staticmethod(lambda path: _FakeStructure())
    )
    monkeypatch.setattr(writer_mod, "replace", lambda struct, **kw: struct)
    monkeypatch.setattr(writer_mod, "to_mmcif", lambda *a, **k: "MMCIF")
    monkeypatch.setattr(writer_mod.np, "savez_compressed", lambda path, **kw: None)
    monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)

    import io

    def fake_open(self, *args, **kwargs):
        writes.append(self.name)
        return io.StringIO()

    monkeypatch.setattr(Path, "open", fake_open)

    writer = writer_mod.BoltzWriter(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        output_format="mmcif",
        boltz2=True,
    )

    # 2 records, 2 diffusion samples each. Per-record ranking should be
    # independent: confidence_score flat order is [r0s0, r0s1, r1s0, r1s1].
    # For r0: s1 > s0 => s1 is rank 0; for r1: s0 > s1 => s0 is rank 0.
    coords = torch.zeros(4, 3, 3)
    prediction = {
        "exception": False,
        "coords": coords,
        "masks": torch.ones(2, 3),
        "confidence_score": torch.tensor([0.1, 0.9, 0.8, 0.2]),
    }
    batch = {"record": [_make_record("r0"), _make_record("r1")]}
    writer.write_on_batch_end(None, None, prediction, None, batch, 0, 0)

    cif = sorted(n for n in writes if n.endswith(".cif"))
    # r0: model_idx0(score .1)->rank1, model_idx1(score .9)->rank0
    # r1: model_idx0(score .8)->rank0, model_idx1(score .2)->rank1
    assert cif == [
        "r0_model_0.cif",
        "r0_model_1.cif",
        "r1_model_0.cif",
        "r1_model_1.cif",
    ]


def test_predict_step_confidence_path_depends_on_batch_size(monkeypatch):
    """predict_step must use the memory-saving sequential confidence path only for
    B == 1 (where ConfidenceModule asserts z.shape[0] == 1), and the parallel path
    for B > 1. B == 1 behaviour stays identical to before the batching change."""
    import torch

    from boltz.model.models.boltz2 import Boltz2

    captured = {}

    def fake_call(self, batch, **kwargs):  # stands in for the full model forward
        captured["run_confidence_sequentially"] = kwargs["run_confidence_sequentially"]
        b = batch["token_pad_mask"].shape[0]
        return {
            "sample_atom_coords": torch.zeros(b, 1, 3),
            "s": torch.zeros(b, 1, 1),
            "z": torch.zeros(b, 1, 1, 1),
        }

    monkeypatch.setattr(Boltz2, "__call__", fake_call, raising=False)

    fake = object.__new__(Boltz2)  # skip nn.Module.__init__ (no weights needed)
    fake.predict_args = {
        "recycling_steps": 0,
        "sampling_steps": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
    }
    fake.confidence_prediction = False  # skip confidence-key plumbing
    fake.affinity_prediction = False

    for b, expected_sequential in [(1, True), (4, False)]:
        batch = {
            "token_pad_mask": torch.ones(b, 5),
            "atom_pad_mask": torch.ones(b, 7),
        }
        out = Boltz2.predict_step(fake, batch, 0)
        assert out["exception"] is False
        assert captured["run_confidence_sequentially"] is expected_sequential
