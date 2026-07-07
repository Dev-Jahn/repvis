#!/usr/bin/env python
"""Offline fidelity harness for the spike/fp8-attention experiment.

Compares two dumps of dense backbone features (an fp16/bf16 baseline vs an fp8
candidate) and the PCA-colored frames they produce, and reports the three gate
metrics from docs/spikes/fp8-attention.md:

  1. per-token feature cosine similarity (mean + worst-case percentiles)
  2. top-3 PCA subspace principal angles (does fp8 rotate the color basis?)
  3. per-pixel color delta-E between the rendered RGB frames

This is deliberately numpy-only and CPU-runnable: it consumes feature/RGB
*dumps*, it does not run any backbone. Produce the dumps on the GPU box, copy
them here, gate offline. `python scripts/fp8_fidelity.py` runs a self-test on
synthetic data (no GPU, no model, no files needed).

Feature dump convention: a .npy of shape (T, H, W, D) float, patch features
BEFORE PCA (i.e. the tensor extract.py returns). RGB dump: (T, H, W, 3) in
[0, 1], the project_chunk output.
"""
from __future__ import annotations

import argparse

import numpy as np


# ---------------------------------------------------------------- metric 1 ---
def feature_cosine(base: np.ndarray, cand: np.ndarray) -> dict:
    """Per-token cosine similarity between two (..., D) feature tensors."""
    a = base.reshape(-1, base.shape[-1]).astype(np.float64)
    b = cand.reshape(-1, cand.shape[-1]).astype(np.float64)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(an * bn, axis=1)
    return {
        "mean": float(cos.mean()),
        "p50": float(np.percentile(cos, 50)),
        "p1": float(np.percentile(cos, 1)),     # worst 1% of tokens
        "min": float(cos.min()),
    }


# ---------------------------------------------------------------- metric 2 ---
def _top_k_basis(feats: np.ndarray, k: int = 3) -> np.ndarray:
    """Top-k right-singular vectors (k, D) of the mean-centered feature matrix —
    the PCA color basis fit() would build."""
    x = feats.reshape(-1, feats.shape[-1]).astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    return vh[:k]


def subspace_principal_angles(base: np.ndarray, cand: np.ndarray, k: int = 3) -> dict:
    """Principal angles (degrees) between the two top-k PCA subspaces.

    Colors are a projection onto the top-3 subspace, so what matters is not
    whether individual components match but whether the *subspace* rotates.
    Angle 0 => identical color basis; a few degrees already visibly remaps hues."""
    qa = _top_k_basis(base, k).T            # (D, k) orthonormal columns
    qb = _top_k_basis(cand, k).T
    s = np.linalg.svd(qa.T @ qb, compute_uv=False).clip(-1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {"angles_deg": [float(a) for a in angles], "max_deg": float(angles.max())}


# ---------------------------------------------------------------- metric 3 ---
def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] (..., 3) -> CIE L*a*b* (D65)."""
    a = rgb.astype(np.float64)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    return np.stack([L, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def color_delta_e(base_rgb: np.ndarray, cand_rgb: np.ndarray) -> dict:
    """Per-pixel CIE76 delta-E (Euclidean in Lab) between two rendered frames.

    CIE76 is the simple gate; swap in CIEDE2000 for the final report — it does
    not change the harness shape. JND ~ 1; delta-E <= 2 is "hard to see",
    >= 5 is "obvious". Feed masked foreground pixels only to match what a viewer
    actually judges."""
    la = _srgb_to_lab(base_rgb.reshape(-1, 3))
    lb = _srgb_to_lab(cand_rgb.reshape(-1, 3))
    de = np.linalg.norm(la - lb, axis=1)
    return {
        "mean": float(de.mean()),
        "p50": float(np.percentile(de, 50)),
        "p95": float(np.percentile(de, 95)),
        "max": float(de.max()),
    }


# ------------------------------------------------------------------- gates ---
GATES = {
    "cosine_mean_min": 0.9995,   # mean per-token cosine must clear this
    "cosine_p1_min": 0.995,      # even the worst 1% of tokens
    "subspace_max_deg": 2.0,     # top-3 color basis may rotate at most this
    "delta_e_p95_max": 3.0,      # 95% of pixels within a small color shift
}


def evaluate(feat_base, feat_cand, rgb_base, rgb_cand) -> dict:
    cos = feature_cosine(feat_base, feat_cand)
    sub = subspace_principal_angles(feat_base, feat_cand)
    de = color_delta_e(rgb_base, rgb_cand)
    verdict = (cos["mean"] >= GATES["cosine_mean_min"]
               and cos["p1"] >= GATES["cosine_p1_min"]
               and sub["max_deg"] <= GATES["subspace_max_deg"]
               and de["p95"] <= GATES["delta_e_p95_max"])
    return {"cosine": cos, "subspace": sub, "delta_e": de, "pass": bool(verdict)}


def _self_test() -> None:
    """CPU self-test: no GPU, no model. Verifies the metrics move in the right
    direction on synthetic 'good' (tiny noise) and 'bad' (rotated basis) fp8s."""
    rng = np.random.default_rng(0)
    T, H, W, D = 4, 16, 16, 64
    base = rng.standard_normal((T, H, W, D)).astype(np.float32)

    # A faithful fp8 candidate: tiny unbiased per-token perturbation.
    good = base + 0.002 * rng.standard_normal(base.shape).astype(np.float32)
    # A harmful one: a structured 5-degree rotation of the leading feature axes
    # (the failure mode that silently swaps colors while cosine stays high).
    th = np.radians(5.0)
    rot = np.eye(D, dtype=np.float32)
    rot[0, 0] = rot[1, 1] = np.cos(th)
    rot[0, 1], rot[1, 0] = -np.sin(th), np.sin(th)
    bad = base @ rot.T

    cg = feature_cosine(base, good)["mean"]
    cb = feature_cosine(base, bad)["mean"]
    ag = subspace_principal_angles(base, good)["max_deg"]
    ab = subspace_principal_angles(base, bad)["max_deg"]
    assert cg > 0.9999, cg
    assert ag < 1.0, ag                       # tiny noise barely rotates basis
    assert ab > 1.0, ab                       # structured rotation is caught...
    assert cb > 0.99, cb                      # ...even though cosine still looks fine

    # delta-E: a small uniform brightness bump should be a small, finite delta-E.
    r0 = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
    de = color_delta_e(r0, np.clip(r0 + 0.02, 0, 1))
    assert 0.5 < de["mean"] < 8.0, de
    print("self-test OK  "
          f"cos(good)={cg:.5f} cos(bad)={cb:.5f} "
          f"angle(good)={ag:.3f}deg angle(bad)={ab:.3f}deg "
          f"deltaE(mean)={de['mean']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-base"); ap.add_argument("--feat-cand")
    ap.add_argument("--rgb-base"); ap.add_argument("--rgb-cand")
    a = ap.parse_args()
    if not any([a.feat_base, a.feat_cand, a.rgb_base, a.rgb_cand]):
        _self_test()
        return
    import json
    res = evaluate(np.load(a.feat_base), np.load(a.feat_cand),
                   np.load(a.rgb_base), np.load(a.rgb_cand))
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
