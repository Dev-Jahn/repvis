"""CPU unit tests for repvis.pca quantile helpers.

    uv run pytest tests/test_pca.py

No GPU needed: everything runs on CPU torch tensors.
"""
import pytest
import torch

from repvis.pca import _weighted_quantile

QS = [0.02, 0.25, 0.5, 0.75, 0.98]
NS = [16, 17, 33, 100]


@pytest.mark.parametrize("n", NS)
def test_equal_weights_match_torch_quantile(n):
    """With equal weights, the weighted type-7 quantile reduces exactly (fp
    tolerance) to torch.quantile's type-7 linear interpolation."""
    torch.manual_seed(n)
    x = torch.randn(n, dtype=torch.float64)
    w = torch.ones(n, dtype=torch.float64)
    q = torch.tensor(QS, dtype=torch.float64)

    got = _weighted_quantile(x, w, QS)
    ref = torch.quantile(x, q)
    assert torch.allclose(got, ref, atol=1e-5, rtol=0.0), (n, got, ref)


def test_small_n_low_q_interpolates_not_clamps():
    """Regression: at n=16, q=0.02 the old midpoint CDF clamped to the min;
    type-7 interpolates between the two smallest sorted values."""
    n = 16
    x = torch.arange(n, dtype=torch.float64)  # sorted 0..15
    w = torch.ones(n, dtype=torch.float64)
    got = _weighted_quantile(x, w, [0.02])
    # type-7: h = 0.02 * (16-1) = 0.30 -> 0 + 0.30*(1-0) = 0.30
    assert torch.isclose(got[0], torch.tensor(0.30, dtype=torch.float64), atol=1e-6)
    assert got[0] > 0.0  # strictly interpolated, not clamped to xs[0]=0


def test_weights_are_monotone_and_bracketed():
    """Non-negative, non-uniform weights: outputs are sorted with q and bracketed
    by the data range."""
    torch.manual_seed(0)
    n = 40
    x = torch.randn(n, dtype=torch.float64)
    w = torch.rand(n, dtype=torch.float64) + 0.1  # strictly positive, non-uniform
    got = _weighted_quantile(x, w, QS)
    assert torch.all(got[1:] >= got[:-1] - 1e-9)      # monotone in q
    assert got.min() >= x.min() - 1e-9
    assert got.max() <= x.max() + 1e-9


def test_degenerate_constant_data():
    """Constant data returns the constant for every quantile under any weights."""
    x = torch.full((8,), 5.0, dtype=torch.float64)
    w = torch.rand(8, dtype=torch.float64) + 0.1
    got = _weighted_quantile(x, w, QS)
    assert torch.allclose(got, torch.full_like(got, 5.0), atol=1e-9)


def test_degenerate_zero_weights_present():
    """Zero weights on some samples stay finite, monotone in q, and bracketed by
    the data range (no NaN/inf from the CDF denominator)."""
    x = torch.tensor([-3.0, 1.0, 5.0, 9.0], dtype=torch.float64)
    w = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    got = _weighted_quantile(x, w, QS)
    assert torch.isfinite(got).all()
    assert torch.all(got[1:] >= got[:-1] - 1e-9)
    assert got.min() >= x.min() - 1e-9 and got.max() <= x.max() + 1e-9
