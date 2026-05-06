from numba import njit, prange
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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


# cargar imagen
image = Image.open("imagenes/penguins_6000x6000.jpg")

# PIL -> NumPy
rgb = np.array(image, dtype=np.float32)

# gris
gray = rgb_to_gray(rgb)

# sobel
result = sobel_parallel(gray)

# mostrar
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
