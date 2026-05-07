from PIL import Image
import numpy as np

from sobel_common import run_cli


def load_image(image_path):
    # cargar imagen
    image = Image.open(image_path)

    # RGB -> numpy
    return np.asarray(image, dtype=np.float32)


def rgb_to_gray_numpy(rgb):
    # convertir a gris
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    return gray_u8


def sobel_numpy(gray_u8):
    # Sobel
    img = gray_u8.astype(np.float32)

    # dimensiones
    height, width = img.shape

    # kernels
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # imagen resultado
    result = np.zeros((height, width), dtype=np.float32)

    gx = (
        Gx[0, 0] * img[:-2, :-2]
        + Gx[0, 1] * img[:-2, 1:-1]
        + Gx[0, 2] * img[:-2, 2:]
        + Gx[1, 0] * img[1:-1, :-2]
        + Gx[1, 1] * img[1:-1, 1:-1]
        + Gx[1, 2] * img[1:-1, 2:]
        + Gx[2, 0] * img[2:, :-2]
        + Gx[2, 1] * img[2:, 1:-1]
        + Gx[2, 2] * img[2:, 2:]
    )

    gy = (
        Gy[0, 0] * img[:-2, :-2]
        + Gy[0, 1] * img[:-2, 1:-1]
        + Gy[0, 2] * img[:-2, 2:]
        + Gy[1, 0] * img[1:-1, :-2]
        + Gy[1, 1] * img[1:-1, 1:-1]
        + Gy[1, 2] * img[1:-1, 2:]
        + Gy[2, 0] * img[2:, :-2]
        + Gy[2, 1] * img[2:, 1:-1]
        + Gy[2, 2] * img[2:, 2:]
    )

    # magnitud gradiente
    result[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)

    return result


def build_runner():
    return {"load": load_image, "gray": rgb_to_gray_numpy, "sobel": sobel_numpy}


if __name__ == "__main__":
    run_cli(build_runner(), "numpy")
