import numpy as np
from skimage.color import rgb2lab
from cvd.engine import simulate


def contrast_preservation(original, candidate, kind, n_pairs=50000, seed=0, min_dist=10):
    orig_lab = rgb2lab(original).reshape(-1, 3)
    seen_lab = rgb2lab(simulate(candidate, kind)).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(orig_lab), n_pairs)
    j = rng.integers(0, len(orig_lab), n_pairs)
    d_orig = np.linalg.norm(orig_lab[i] - orig_lab[j], axis=1)
    d_seen = np.linalg.norm(seen_lab[i] - seen_lab[j], axis=1)
    keep = d_orig > min_dist
    if keep.sum() == 0:
        return 1.0
    return float((d_seen[keep] / d_orig[keep]).mean())


def naturalness(original, corrected):
    return float(np.linalg.norm(rgb2lab(original) - rgb2lab(corrected), axis=2).mean())


def figure_contrast(image, reference, kind, mask, valid_light_cut=90, n_pairs=50000, seed=0):
    valid = rgb2lab(reference)[:, :, 0] < valid_light_cut
    fig = np.flatnonzero((mask & valid).ravel())
    bg = np.flatnonzero((~mask & valid).ravel())
    if len(fig) == 0 or len(bg) == 0:
        return 0.0
    seen = rgb2lab(simulate(image, kind)).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    a = rng.choice(fig, n_pairs)
    b = rng.choice(bg, n_pairs)
    return float(np.linalg.norm(seen[a] - seen[b], axis=1).mean())