"""CPU unit test for the _auto_seed artifact-token filter.

    uv run pytest tests/test_autoseed.py

No GPU needed: _auto_seed is pure tensor math on a frame-0 feature grid.
"""
import os
import tempfile

# Redirect sources/ + runs/ away from the repo before repvis.config is imported.
os.environ.setdefault("REPVIS_DATA_DIR", tempfile.mkdtemp(prefix="repvis-test-"))

import torch  # noqa: E402

from repvis.pipeline import _auto_seed  # noqa: E402


def test_norm_outlier_token_never_seeds_the_primary_point():
    """A gross high-norm artifact token (register-less ViT) must be excluded from
    the positive-peak candidates: the primary (+) prompt lands on the object blob,
    not on the artifact patch, and no positive is planted on the artifact cell.
    Mimics the measured regime: background norms with realistic spread (MAD > 0),
    object norms inside the |z| <= 3.5 band, the artifact far outside it."""
    gh = gw = 8
    D = 16
    base = torch.zeros(D)
    base[0] = 1.0                                   # background direction
    obj = torch.zeros(D)
    obj[1] = 1.0                                    # object direction
    grid = torch.empty(gh, gw, D)
    for r in range(gh):                             # flat-ish background with a
        for c in range(gw):                         # deterministic norm spread
            grid[r, c] = base * (1.0 + 0.02 * ((r * gw + c) % 11 - 5))
    for r, c in ((3, 3), (3, 4), (4, 3), (4, 4)):   # object blob: distinct
        grid[r, c] = obj * 1.2                      # DIRECTION, norm in-band
    grid[0, 7] = base * 10.0                        # artifact: gross norm outlier

    W, H = 640, 360
    pts = _auto_seed(grid, W, H)

    def cell(px, py):
        return int(py / H * gh), int(px / W * gw)

    pos = [(x, y) for (x, y, lab, _f) in pts if lab == 1]
    neg = [(x, y) for (x, y, lab, _f) in pts if lab == 0]
    assert pos and len(neg) == 1
    # the primary peak sits on the object blob, not on the artifact
    assert cell(*pos[0]) in {(3, 3), (3, 4), (4, 3), (4, 4)}
    # no positive prompt lands on the artifact cell
    assert all(cell(x, y) != (0, 7) for (x, y) in pos)


def test_uniform_frame_negative_never_shares_cell_with_a_positive():
    """Degenerate exactly-uniform frame: every patch is identical, so saliency is
    flat (MAD = 0, nothing flagged), the positive argmax picks cell (0, 0), and the
    least-salient border cell (border argmin) is ALSO (0, 0). A single grid cell must
    never carry contradictory (+) and (-) prompts, so the border negative must skip
    any cell already holding a positive and land on a positive-free border cell (or,
    if none is free, be dropped). Guards the same-cell collision only — the negative
    staying otherwise ungated (allowed on a nearby cell / on the subject) is unchanged."""
    gh = gw = 8
    D = 16
    grid = torch.ones(gh, gw, D)                    # exactly uniform: flat, MAD = 0

    W, H = 640, 360
    pts = _auto_seed(grid, W, H)

    def cell(px, py):
        return int(py / H * gh), int(px / W * gw)

    pos = {cell(x, y) for (x, y, lab, _f) in pts if lab == 1}
    neg = {cell(x, y) for (x, y, lab, _f) in pts if lab == 0}
    assert pos                                      # a primary positive is always planted
    assert pos & neg == set()                       # no cell carries both (+) and (-)
