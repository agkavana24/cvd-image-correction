import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_erosion, label, uniform_filter
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu


def load_image(path, max_size=1000):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    return np.asarray(img, dtype=np.float64) / 255.0


def save_image(arr, path):
    arr = np.clip(arr, 0, 1)
    Image.fromarray(np.rint(arr * 255).astype("uint8")).save(path)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
LMS2RGB = np.linalg.inv(RGB2LMS)

SIM = {
    "protanopia":   np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    "deuteranopia": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
    "tritanopia":   np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
}

SHIFT = np.array([[0.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])


def rgb_sim_matrix(kind):
    return LMS2RGB @ SIM[kind] @ RGB2LMS


def apply_matrix(img, M):
    h, w, _ = img.shape
    return (img.reshape(-1, 3) @ M.T).reshape(h, w, 3)


def simulate(img_srgb, kind):
    lin = srgb_to_linear(img_srgb)
    return linear_to_srgb(apply_matrix(lin, rgb_sim_matrix(kind)))


def null_space_info(kind):
    S = rgb_sim_matrix(kind)
    rank = np.linalg.matrix_rank(S, tol=1e-6)
    U, s, Vt = np.linalg.svd(S)
    null_dir = Vt[-1]
    return rank, s, null_dir, S @ null_dir


def correct(img_srgb, kind, alpha=1.0, shift=None):
    if shift is None:
        shift = SHIFT
    lin = srgb_to_linear(img_srgb)
    sim = apply_matrix(lin, rgb_sim_matrix(kind))
    lost = lin - sim
    shifted = apply_matrix(lost, shift)
    return linear_to_srgb(lin + alpha * shifted)


LIGHT_CUT = 90
CANDIDATE_COLOURS = [
    ("dark blue", (0.10, 0.10, 0.60)),
    ("bright yellow", (1.00, 0.90, 0.10)),
    ("white", (1.00, 1.00, 1.00)),
]


def detect_figure(original):
    lab = rgb2lab(original)
    L = lab[:, :, 0]
    valid = L < LIGHT_CUT
    info = {"confident": False, "fraction": 0.0, "separation": 0.0}
    empty = np.zeros(L.shape, dtype=bool)

    core = binary_erosion(valid, iterations=1 if min(L.shape) < 500 else 2)
    if core.sum() < 500:
        core = valid
    pts = lab[:, :, 1:3][core]
    if pts.shape[0] < 500:
        return empty, info

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    info["explained"] = float(eigvals.max() / eigvals.sum())

    proj = centered @ axis
    if np.ptp(proj) < 1e-6:
        return empty, info
    t = threshold_otsu(proj)
    high = proj > t
    n_high, n_low = int(high.sum()), int((~high).sum())
    if min(n_high, n_low) == 0:
        return empty, info

    figure_is_high = n_high < n_low
    fig_vals = proj[high] if figure_is_high else proj[~high]
    bg_vals = proj[~high] if figure_is_high else proj[high]
    pooled = np.sqrt((fig_vals.var() + bg_vals.var()) / 2) + 1e-9
    separation = abs(fig_vals.mean() - bg_vals.mean()) / pooled
    fraction = len(fig_vals) / len(proj)

    eta = float(len(fig_vals) * len(bg_vals) / len(proj) ** 2
                * (fig_vals.mean() - bg_vals.mean()) ** 2 / proj.var())
    components, _ = label(valid)
    sizes = np.bincount(components.ravel())[1:]
    n_dots = int((sizes >= 3).sum())

    labels = np.zeros(L.shape, dtype=bool)
    full_proj = (lab[:, :, 1:3] - mean) @ axis
    side = full_proj > t
    labels[valid] = (side if figure_is_high else ~side)[valid]

    h, w = L.shape
    win = int(max(5, min(h, w) // 32)) | 1
    num = uniform_filter((labels & valid).astype(float), size=win)
    den = uniform_filter(valid.astype(float), size=win)
    mask = (num / (den + 1e-9) > 0.5) & valid

    info.update(separation=float(separation), fraction=float(fraction), eta=eta, n_dots=n_dots,
                confident=bool(separation > 1.8 and 0.03 < fraction < 0.5
                               and eta >= 0.7 and n_dots >= 25))
    return mask, info


def _viewer_lab(rgb, kind):
    rgb = np.asarray(rgb, dtype=float).reshape(-1, 1, 3)
    return rgb2lab(simulate(rgb, kind)).reshape(-1, 3)


def choose_figure_colour(original, mask, kind):
    valid = rgb2lab(original)[:, :, 0] < LIGHT_CUT
    bg = valid & ~mask
    if bg.sum() == 0:
        return CANDIDATE_COLOURS[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(bg.ravel()), size=min(4000, int(bg.sum())), replace=False)
    bg_lab = rgb2lab(simulate(original, kind)).reshape(-1, 3)[idx]
    scores = []
    for name, rgb in CANDIDATE_COLOURS:
        cand = _viewer_lab(rgb, kind)[0]
        dist = np.linalg.norm(bg_lab - cand, axis=1)
        scores.append(np.percentile(dist, 10))
    best = int(np.argmax(scores))
    if scores[0] >= 0.8 * scores[best]:
        best = 0
    return CANDIDATE_COLOURS[best]


def emphasize_figure(corrected, original, kind="deuteranopia"):
    mask, info = detect_figure(original)
    if not info["confident"]:
        return corrected, mask, info
    name, colour = choose_figure_colour(original, mask, kind)
    out = corrected.copy()
    out[mask] = np.array(colour)
    info["colour_name"] = name
    return out, mask, info

# --------------------------------------------------------------------------
# Universal contrast-boosting correction (works on any image, any deficiency)
# --------------------------------------------------------------------------


def _seen_spread(rgb, kind, valid, n=20000, seed=0):
    lab = rgb2lab(simulate(rgb, kind)).reshape(-1, 3)
    idx = np.flatnonzero(valid.ravel())
    rng = np.random.default_rng(seed)
    a = rng.choice(idx, n)
    b = rng.choice(idx, n)
    return float(np.linalg.norm(lab[a] - lab[b], axis=1).mean())


def correct_universal(img_srgb, kind, alpha=1.0, strength_L=30.0, strength_C=40.0):
    import warnings
    from skimage.color import lab2rgb

    lab = rgb2lab(img_srgb)
    seen = rgb2lab(simulate(img_srgb, kind))
    lost = lab - seen

    dots = lab[:, :, 0] < LIGHT_CUT
    use = dots if dots.sum() > 100 else np.ones_like(dots)

    flat = lost[use]
    centered = flat - flat.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, -1]

    c = lost @ u
    c0 = np.median(c[use])
    scale = max(np.percentile(np.abs(c[use] - c0), 90), 8.0)
    cn = np.tanh((c - c0) / scale)
    cn = np.where(dots, cn, 0.0)

    axis = 1 if kind == "tritanopia" else 2

    def build(lab_in, cn_in, sL, sC):
        out = lab_in.copy()
        out[:, :, 0] = np.clip(lab_in[:, :, 0] + sL * alpha * strength_L * cn_in, 0, 100)
        out[:, :, axis] = lab_in[:, :, axis] + sC * alpha * strength_C * cn_in
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.clip(lab2rgb(out), 0, 1)

    lab_s, cn_s, dots_s = lab[::3, ::3], cn[::3, ::3], dots[::3, ::3]
    valid_s = dots_s if dots_s.sum() > 100 else np.ones_like(dots_s)
    best, best_score = (1, 1), -1.0
    for sL in (1, -1):
        for sC in (1, -1):
            score = _seen_spread(build(lab_s, cn_s, sL, sC), kind, valid_s)
            if score > best_score:
                best, best_score = (sL, sC), score
    return build(lab, cn, best[0], best[1])


def correct_best(img_srgb, kind, alpha=1.0):
    candidates = [
        correct_universal(img_srgb, kind, alpha=alpha),
        correct_universal(correct(img_srgb, kind, alpha=alpha), kind, alpha=alpha),
    ]
    lab_orig = rgb2lab(img_srgb[::3, ::3])
    valid = lab_orig[:, :, 0] < LIGHT_CUT
    if valid.sum() < 100:
        valid = np.ones_like(valid)
    best, best_score = candidates[0], -1e9
    for cand in candidates:
        small = cand[::3, ::3]
        spread = _seen_spread(small, kind, valid)
        drift = float(np.linalg.norm(rgb2lab(small) - lab_orig, axis=2).mean())
        score = spread - 0.3 * drift
        if score > best_score:
            best, best_score = cand, score
    return best