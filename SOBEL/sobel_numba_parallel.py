from numba import get_num_threads, njit, prange
from PIL import Image
import numpy as np

from sobel_common import run_cli


@njit(parallel=True)
def rgb_to_gray(rgb):

    height, width, _ = rgb.shape

    gray = np.zeros((height, width), dtype=np.float32)

    for i in prange(height):

        for j in range(width):

            r = rgb[i, j, 0]
            g = rgb[i, j, 1]
            b = rgb[i, j, 2]

            i_gray = 0.299 * r + 0.587 * g + 0.114 * b

            if i_gray < 0:
                i_gray = 0
            elif i_gray > 255:
                i_gray = 255

            gray[i, j] = i_gray

    return gray


@njit(parallel=True)
def sobel_parallel(gray):

    height, width = gray.shape

    kernel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )

    kernel_y = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )

    result = np.zeros((height, width), dtype=np.float32)

    for row in prange(1, height - 1):

        for col in range(1, width - 1):

            gx = 0.0
            gy = 0.0

            for kr in range(3):

                for kc in range(3):

                    pixel = gray[row + kr - 1, col + kc - 1]

                    gx += pixel * kernel_x[kr, kc]
                    gy += pixel * kernel_y[kr, kc]

            result[row, col] = abs(gx) + abs(gy)

    return result


def build_runner():
    def load_step(image_path):
        # cargar imagen
        image = Image.open(image_path)

        # PIL -> NumPy
        return np.array(image, dtype=np.float32)

    def gray_step(rgb):
        # gris
        return rgb_to_gray(rgb)

    def warmup(image_path):
        image = Image.open(image_path).resize((8, 8))
        rgb = np.array(image, dtype=np.float32)
        gray = rgb_to_gray(rgb)
        sobel_parallel(gray)

    def info():
        return {"Workers Numba": get_num_threads()}

    return {
        "load": load_step,
        "gray": gray_step,
        "sobel": sobel_parallel,
        "warmup": warmup,
        "info": info,
    }


if __name__ == "__main__":
    run_cli(build_runner(), "numba_parallel")
