import numpy as np
from PIL import Image


def load_image(path, max_size=1000):
       img = Image.open(path).convert("RGB")
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
       check = S @ null_dir
       return rank, s, null_dir, check