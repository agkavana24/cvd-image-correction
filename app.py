import os
import uuid
import time
import glob

from flask import Flask, render_template, request

from cvd.engine import load_image, save_image, simulate, correct_best, emphasize_figure
from cvd.scoring import contrast_preservation, naturalness, figure_contrast

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

OUT = os.path.join(app.static_folder, "outputs")
os.makedirs(OUT, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg"}
KINDS = ("protanopia", "deuteranopia", "tritanopia")


def cleanup_old_outputs(max_age_seconds=3600):
    now = time.time()
    for path in glob.glob(os.path.join(OUT, "*.png")):
        try:
            if now - os.path.getmtime(path) > max_age_seconds:
                os.remove(path)
        except OSError:
            pass


def show_error(message, status=400):
    return render_template("index.html", error=message), status


@app.route("/")
def index():
    return render_template("index.html", error=None)


@app.errorhandler(413)
def too_large(_):
    return show_error("That file is larger than 5 MB. Please upload a smaller image.", 413)


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    kind = request.form.get("kind", "deuteranopia")
    highlight = request.form.get("highlight") == "on"

    try:
        alpha = float(request.form.get("alpha", 1.0))
    except ValueError:
        return show_error("Invalid strength value.")
    alpha = min(max(alpha, 0.0), 3.0)

    if not file or file.filename == "":
        return show_error("Please choose an image first.")
    if "." not in file.filename or file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return show_error("Please upload a PNG or JPG image.")
    if kind not in KINDS:
        return show_error("Invalid deficiency type.")

    try:
        img = load_image(file)
    except Exception:
        return show_error("Could not read that image file. Is it a valid PNG or JPG?")

    cleanup_old_outputs()

    fixed = correct_best(img, kind, alpha=alpha)

    mask = None
    figure_info = None
    fig_before = fig_after = None
    if highlight:
        fixed, mask, figure_info = emphasize_figure(fixed, img, kind)
        if figure_info["confident"]:
            fig_before = round(figure_contrast(img, img, kind, mask), 1)
            fig_after = round(figure_contrast(fixed, img, kind, mask), 1)

    base = contrast_preservation(img, img, kind)
    score = contrast_preservation(img, fixed, kind)
    drift = naturalness(img, fixed)

    uid = uuid.uuid4().hex[:8]
    images = {
        "original": img,
        "before": simulate(img, kind),
        "corrected": fixed,
        "after": simulate(fixed, kind),
    }
    if mask is not None and figure_info["confident"]:
        images["mask"] = mask.astype(float)[:, :, None].repeat(3, axis=2)

    paths = {}
    for name, arr in images.items():
        fname = f"{uid}_{name}.png"
        save_image(arr, os.path.join(OUT, fname))
        paths[name] = f"outputs/{fname}"

    return render_template(
        "result.html",
        paths=paths,
        kind=kind,
        alpha=alpha,
        highlight=highlight,
        figure_info=figure_info,
        fig_before=fig_before,
        fig_after=fig_after,
        base=round(base, 3),
        score=round(score, 3),
        pct=round((score - base) / base * 100, 1),
        drift=round(drift, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)