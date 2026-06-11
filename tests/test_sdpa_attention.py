"""Tests for the SDPA-based AttentionPairBias forward pass.

These compare the new ``scaled_dot_product_attention`` path against the
original hand-rolled einsum formula (kept intact behind ``sdpa=False``).
"""

import torch

from boltz.model.layers.attentionv2 import AttentionPairBias


def _make_module(c_s=16, c_z=8, num_heads=4, compute_pair_bias=True, sdpa=True):
    torch.manual_seed(0)
    return AttentionPairBias(
        c_s=c_s,
        c_z=c_z,
        num_heads=num_heads,
        compute_pair_bias=compute_pair_bias,
        sdpa=sdpa,
    )


def _make_inputs(B=2, N=7, c_s=16, c_z=8, dtype=torch.float32, all_mask_row=False):
    torch.manual_seed(1)
    s = torch.randn(B, N, c_s, dtype=dtype)
    z = torch.randn(B, N, N, c_z, dtype=dtype)
    mask = (torch.rand(B, N) > 0.3).to(dtype)
    if all_mask_row:
        # Make one full sample fully masked (all key positions masked) to
        # exercise the "fully-masked row" branch without NaNs.
        mask[0] = 0.0
    else:
        # Guarantee at least one unmasked key per sample so the reference
        # softmax over keys is well-defined for the non-degenerate case.
        mask[:, 0] = 1.0
    return s, z, mask


def _clone_weights(dst, src):
    dst.load_state_dict(src.state_dict())


def _build_pair(sdpa_kwargs, ein_kwargs, **mod_kwargs):
    """Build two modules with identical weights, one SDPA, one einsum."""
    m_sdpa = _make_module(sdpa=True, **mod_kwargs)
    m_ein = _make_module(sdpa=False, **mod_kwargs)
    _clone_weights(m_ein, m_sdpa)
    m_sdpa.eval()
    m_ein.eval()
    return m_sdpa, m_ein


def test_sdpa_matches_einsum_fp32():
    m_sdpa, m_ein = _build_pair({}, {})
    s, z, mask = _make_inputs()
    with torch.no_grad():
        out_new = m_sdpa(s, z, mask, k_in=s)
        out_ref = m_ein(s, z, mask, k_in=s)
    assert out_new.shape == out_ref.shape
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_new, out_ref, atol=1e-4, rtol=1e-4)


def test_sdpa_matches_einsum_fp32_precomputed_bias():
    # compute_pair_bias=False: module receives a precomputed per-head bias z.
    num_heads = 4
    m_sdpa, m_ein = _build_pair({}, {}, compute_pair_bias=False, num_heads=num_heads)
    B, N, c_s = 2, 7, 16
    torch.manual_seed(2)
    s = torch.randn(B, N, c_s)
    # precomputed additive bias, shape (B, N, N, H) -> Rearrange to (B, H, N, N)
    z = torch.randn(B, N, N, num_heads)
    mask = (torch.rand(B, N) > 0.3).float()
    mask[:, 0] = 1.0
    with torch.no_grad():
        out_new = m_sdpa(s, z, mask, k_in=s)
        out_ref = m_ein(s, z, mask, k_in=s)
    assert torch.allclose(out_new, out_ref, atol=1e-4, rtol=1e-4)


def test_sdpa_matches_einsum_bf16():
    m_sdpa, m_ein = _build_pair({}, {})
    m_sdpa = m_sdpa.bfloat16()
    m_ein = m_ein.bfloat16()
    s, z, mask = _make_inputs(dtype=torch.float32)
    s = s.bfloat16()
    z = z.bfloat16()
    with torch.no_grad():
        out_new = m_sdpa(s.clone(), z.clone(), mask, k_in=s.clone())
        out_ref = m_ein(s.clone(), z.clone(), mask, k_in=s.clone())
    assert out_new.dtype == torch.bfloat16
    assert torch.isfinite(out_new.float()).all()
    # Looser tol documents FP behavior; internal math is fp32 in both paths.
    assert torch.allclose(out_new.float(), out_ref.float(), atol=2e-2, rtol=2e-2)


def test_sdpa_fully_masked_row_no_nan():
    m_sdpa, m_ein = _build_pair({}, {})
    s, z, mask = _make_inputs(all_mask_row=True)
    with torch.no_grad():
        out_new = m_sdpa(s, z, mask, k_in=s)
        out_ref = m_ein(s, z, mask, k_in=s)
    assert torch.isfinite(out_new).all(), "SDPA output contains NaN/Inf"
    # Both paths use a large finite negative, so degenerate rows behave the same.
    assert torch.allclose(out_new, out_ref, atol=1e-4, rtol=1e-4)
