"""Tests for the opt/compile-score branch.

These cover the two mechanisms this branch adds:

(a) COMPILE SANITY: torch.compile(..., dynamic=False, fullgraph=False) of a
    small stand-in score net produces outputs matching eager within tolerance,
    and re-running with the same (padded) input shape does not error.

(b) PADDING CORRECTNESS: the real ``boltz.data.pad`` helpers pad samples of
    different lengths to a fixed size, the pad mask marks the padded positions,
    and the unpadded region is left untouched.

CPU only, no model download. The compile path exercises the same flags
(``dynamic=False, fullgraph=False``) used in
``boltz/model/modules/diffusionv2.py``.
"""

import torch
from torch import nn

from boltz.data.pad import pad_dim, pad_to_max


# ---------------------------------------------------------------------------
# (a) COMPILE SANITY
# ---------------------------------------------------------------------------


class _TinyScoreNet(nn.Module):
    """A tiny stand-in for the diffusion score network.

    A 2-layer MLP plus a single self-attention block. This is enough to prove
    that torch.compile with the production flags works on this torch build and
    that recompile-free re-runs at a fixed shape succeed.
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        return self.mlp(x + a)


def test_compile_matches_eager_and_reruns():
    torch._dynamo.reset()
    torch.manual_seed(0)

    dim = 16
    model = _TinyScoreNet(dim=dim).eval()

    # Fixed padded shape: (batch, tokens, dim).
    x = torch.randn(2, 8, dim)

    with torch.no_grad():
        eager_out = model(x)

    compiled = torch.compile(model, dynamic=False, fullgraph=False)

    with torch.no_grad():
        compiled_out_1 = compiled(x)
        # Re-run with the SAME shape: must not error (no recompile expected).
        compiled_out_2 = compiled(x.clone())

    assert compiled_out_1.shape == eager_out.shape
    assert torch.allclose(compiled_out_1, eager_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(compiled_out_1, compiled_out_2, atol=1e-5, rtol=1e-5)


def test_compile_fixed_shape_no_error_across_calls():
    """Two independent samples padded to the same fixed shape both run."""
    torch._dynamo.reset()
    torch.manual_seed(1)

    dim = 16
    model = _TinyScoreNet(dim=dim).eval()
    compiled = torch.compile(model, dynamic=False, fullgraph=False)

    fixed_tokens = 12
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(1, fixed_tokens, dim)
            out = compiled(x)
            assert out.shape == (1, fixed_tokens, dim)


# ---------------------------------------------------------------------------
# (b) PADDING CORRECTNESS
# ---------------------------------------------------------------------------


def test_pad_to_max_marks_pad_and_preserves_data():
    """pad_to_max pads to the in-batch max; mask marks padding, data intact."""
    # Two samples of different "token" lengths, feature dim 3.
    a = torch.arange(5 * 3, dtype=torch.float).reshape(5, 3)
    b = torch.arange(8 * 3, dtype=torch.float).reshape(8, 3) + 100.0

    data, mask = pad_to_max([a, b], value=0)

    # Stacked, padded to the max length (8).
    assert data.shape == (2, 8, 3)
    # mask is non-trivial (a tensor) because the shapes differed.
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (2, 8, 3)

    # Unpadded region equals the original tensors (no corruption).
    assert torch.equal(data[0, :5], a)
    assert torch.equal(data[1, :8], b)

    # Padded region of the shorter sample is zero.
    assert torch.all(data[0, 5:] == 0)

    # Mask marks real positions as 1 and padded positions as 0.
    assert torch.all(mask[0, :5] == 1)
    assert torch.all(mask[0, 5:] == 0)
    assert torch.all(mask[1] == 1)  # b is already at max length


def test_pad_dim_fixed_target_and_mask():
    """pad_dim to a fixed target keeps original values and zero-pads the rest.

    This mirrors how the featurizer extends ``token_pad_mask`` /
    ``atom_pad_mask`` (an all-ones vector) to a fixed size: padding with 0
    correctly marks the new positions as padding.
    """
    fixed_tokens = 10

    feat = torch.arange(6, dtype=torch.float)  # length-6 "feature"
    pad_mask = torch.ones(6, dtype=torch.float)  # all real

    pad_len = fixed_tokens - feat.shape[0]
    feat_p = pad_dim(feat, 0, pad_len)
    mask_p = pad_dim(pad_mask, 0, pad_len)  # value defaults to 0

    assert feat_p.shape[0] == fixed_tokens
    assert mask_p.shape[0] == fixed_tokens

    # Real region preserved.
    assert torch.equal(feat_p[:6], feat)
    # Padding is zero in both feature and mask.
    assert torch.all(feat_p[6:] == 0)
    assert torch.all(mask_p[:6] == 1)
    assert torch.all(mask_p[6:] == 0)


def test_pad_to_max_same_shape_returns_zero_mask():
    """When all samples already share a shape, pad_to_max returns mask=0."""
    a = torch.randn(4, 3)
    b = torch.randn(4, 3)
    data, mask = pad_to_max([a, b], value=0)
    assert data.shape == (2, 4, 3)
    assert mask == 0  # sentinel: no padding was applied
