import numpy as np
from PIL import Image

def load_image(path, max_size=1000):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return np.asarray(img, dtype=np.float64) / 255.0

def save_image(arr, path):
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype("uint8")).save(path)

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)