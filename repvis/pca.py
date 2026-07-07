"""PCA -> RGB rendering of dense patch features (all on GPU).

Split into a fit step (on a capped token subsample gathered while streaming) and
a per-chunk projection step, so we never materialize all features at once.

Foreground/background separation is NOT done here anymore: it is handled by SAM2
pixel-accurate masks (see `sam.py` / `pipeline.py`), applied to the rendered
frames. This module only maps features to colors. The per-cell "Refit" re-fits
the display basis over the foreground tokens (`refit_display`) so within-subject
color contrast stands out.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


def _subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    return x[idx]


@torch.inference_mode()
def _fit(x: torch.Tensor, k: int, fit_max: int):
    """Return top-k principal components (k, D) and the mean (D,)."""
    xs = _subsample(x, fit_max)
    mean = xs.mean(0)
    xc = xs - mean
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = vh[:k].clone()
    for i in range(k):  # deterministic sign -> stable colors run-to-run
        j = torch.argmax(comps[i].abs())
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return comps, mean


@torch.inference_mode()
def _fit_weighted(x: torch.Tensor, w: torch.Tensor, k: int):
    """Weighted top-k principal components (k, D) and weighted mean (D,).

    `w` are per-token weights (>= 0, aligned to `x`). Weighted mean
    `mu = sum(w_i x_i) / sum(w_i)`, then SVD on `sqrt(w)*(x-mu)` so each token
    contributes to the basis in proportion to its weight."""
    wsum = w.sum().clamp_min(1e-12)
    mean = (w[:, None] * x).sum(0) / wsum
    xc = w.sqrt()[:, None] * (x - mean)
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = vh[:k].clone()
    for i in range(k):  # deterministic sign -> stable colors run-to-run
        j = torch.argmax(comps[i].abs())
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return comps, mean


@torch.inference_mode()
def _quantile(x: torch.Tensor, qs) -> torch.Tensor:
    x = _subsample(x, 200_000)
    return torch.quantile(x, torch.tensor(qs, device=x.device, dtype=x.dtype))


@torch.inference_mode()
def _weighted_quantile(x: torch.Tensor, w: torch.Tensor, qs) -> torch.Tensor:
    """Weighted quantiles of 1-D `x` (non-negative weights `w`, aligned to `x`).

    `qs` are fractions in [0, 1]. Linear interpolation on the weighted CDF using
    the plotting position `p_i = below_i / (below_i + above_i)`, where `below_i`
    (`above_i`) is the total weight strictly below (above) sorted sample `i`. This
    is the weighted generalization of `torch.quantile`'s type-7 rule `i/(n-1)`,
    to which it reduces exactly when weights are equal. The plotting position is
    monotone non-decreasing for non-negative weights."""
    order = torch.argsort(x)
    xs, ws = x[order], w[order]
    cw = torch.cumsum(ws, 0)
    below = cw - ws                                # weight strictly below sample i
    cdf = below / (cw[-1] - ws).clamp_min(1e-12)   # below / (below + above)
    q = torch.as_tensor(qs, device=x.device, dtype=x.dtype)
    hi = torch.searchsorted(cdf, q).clamp(max=xs.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    t = ((q - cdf[lo]) / (cdf[hi] - cdf[lo]).clamp_min(1e-12)).clamp(0.0, 1.0)
    out = xs[lo] + t * (xs[hi] - xs[lo])
    out = torch.where(q <= cdf[0], xs[0], out)
    return torch.where(q >= cdf[-1], xs[-1], out)


@dataclass
class PCAState:
    l2norm: bool
    mean: torch.Tensor    # (D,)
    comps: torch.Tensor   # (3, D)
    lo: torch.Tensor      # (3,)
    hi: torch.Tensor      # (3,)

    def to(self, device) -> "PCAState":
        """Non-mutating: returns a copy on `device` (states are shared across
        concurrent render threads, one per GPU)."""
        return PCAState(self.l2norm, self.mean.to(device), self.comps.to(device),
                        self.lo.to(device), self.hi.to(device))


@torch.inference_mode()
def fit_pca_state(fit_tokens: torch.Tensor, *, l2norm: bool = False,
                  percentiles=(2.0, 98.0)) -> PCAState:
    """Fit the PCA basis + display range from a (M, D) token sample (capped upstream)."""
    x = fit_tokens.float()
    if l2norm:
        x = F.normalize(x, dim=1)
    comps, mean = _fit(x, 3, x.shape[0])
    z = (x - mean) @ comps.T
    p = [percentiles[0] / 100.0, percentiles[1] / 100.0]
    lo = torch.empty(3, device=x.device)
    hi = torch.empty(3, device=x.device)
    for c in range(3):
        q = _quantile(z[:, c], p)
        lo[c], hi[c] = q[0], q[1]
    return PCAState(l2norm, mean, comps, lo, hi)


@torch.inference_mode()
def project_chunk(feats: torch.Tensor, st: PCAState) -> torch.Tensor:
    """feats (T,H,W,D) -> rgb (T,H,W,3) in [0,1] using a fixed basis. Unmasked —
    the SAM foreground mask is multiplied in at render time (pipeline)."""
    T, H, W, D = feats.shape
    x = feats.reshape(-1, D).float()
    disp = F.normalize(x, dim=1) if st.l2norm else x
    z = (disp - st.mean) @ st.comps.T
    rgb = ((z - st.lo) / (st.hi - st.lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return rgb.reshape(T, H, W, 3)


@torch.inference_mode()
def refit_display(fg_tokens: torch.Tensor, st: PCAState, *,
                  weights: torch.Tensor | None = None,
                  percentiles=(2.0, 98.0)) -> PCAState:
    """Re-fit the display basis (mean/comps/lo/hi) on `fg_tokens` — the per-cell
    'Refit' that maximizes color contrast within the current foreground.

    With per-token `weights` in [0, 1] (e.g. the per-grid-token foreground
    fraction) the fit is a WEIGHTED PCA: weighted mean, SVD on `sqrt(w)*(x-mean)`,
    and weighted projected quantiles for lo/hi. Thin structures (small but nonzero
    weight) then survive the refit proportionally instead of being hard-gated out.
    No weights -> the original unweighted behavior."""
    x = fg_tokens.float()
    if st.l2norm:
        x = F.normalize(x, dim=1)
    if weights is None:
        comps, mean = _fit(x, 3, x.shape[0])
    else:
        weights = weights.to(device=x.device, dtype=x.dtype).clamp_min(0)
        comps, mean = _fit_weighted(x, weights, 3)
    z = (x - mean) @ comps.T
    p = [percentiles[0] / 100.0, percentiles[1] / 100.0]
    lo = torch.empty(3, device=x.device)
    hi = torch.empty(3, device=x.device)
    for c in range(3):
        q = _quantile(z[:, c], p) if weights is None else _weighted_quantile(z[:, c], weights, p)
        lo[c], hi[c] = q[0], q[1]
    return replace(st, mean=mean, comps=comps, lo=lo, hi=hi)
