import math
from dataclasses import dataclass

import numpy as np
from numba import cuda
from PIL import Image

from sobel_common import run_cli

THREADS_PER_BLOCK = (16, 16)

GX_KERNEL = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1),
)
GY_KERNEL = (
    (1, 2, 1),
    (0, 0, 0),
    (-1, -2, -1),
)


@dataclass
class CudaGrayImage:
    gray_device: cuda.cudadrv.devicearray.DeviceNDArray
    blocks_per_grid: tuple[int, int]


@cuda.jit
def rgb_to_gray_cuda(rgb, gray):
    y, x = cuda.grid(2)
    height, width, _ = rgb.shape

    if y >= height or x >= width:
        return

    r = float(rgb[y, x, 0])
    g = float(rgb[y, x, 1])
    b = float(rgb[y, x, 2])

    gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
    gray[y, x] = 0 if gray_value < 0 else 255 if gray_value > 255 else gray_value


@cuda.jit
def sobel_cuda(gray, out):
    y, x = cuda.grid(2)
    height, width = gray.shape

    if y >= height or x >= width:
        return

    if y == 0 or x == 0 or y == height - 1 or x == width - 1:
        out[y, x] = 0
        return

    gx = 0
    gy = 0

    for ky in range(3):
        for kx in range(3):
            pixel = int(gray[y + ky - 1, x + kx - 1])
            gx += pixel * GX_KERNEL[ky][kx]
            gy += pixel * GY_KERNEL[ky][kx]

    magnitude = int(math.sqrt(gx * gx + gy * gy))
    out[y, x] = 255 if magnitude > 255 else magnitude


def blocks_for_shape(height, width):
    blocks_y = (height + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
    blocks_x = (width + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
    return blocks_y, blocks_x


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def rgb_to_gray(rgb):
    height, width, _ = rgb.shape
    blocks_per_grid = blocks_for_shape(height, width)

    rgb_device = cuda.to_device(rgb)
    gray_device = cuda.device_array((height, width), dtype=np.uint8)

    rgb_to_gray_cuda[blocks_per_grid, THREADS_PER_BLOCK](rgb_device, gray_device)
    cuda.synchronize()

    return CudaGrayImage(gray_device=gray_device, blocks_per_grid=blocks_per_grid)


def sobel_numba_cuda(gray_image):
    height, width = gray_image.gray_device.shape
    sobel_device = cuda.device_array((height, width), dtype=np.uint8)

    sobel_cuda[gray_image.blocks_per_grid, THREADS_PER_BLOCK](
        gray_image.gray_device,
        sobel_device,
    )

    cuda.synchronize()

    return sobel_device.copy_to_host()


def warmup(image_path):
    image = Image.open(image_path).convert("RGB").resize((8, 8))
    rgb = np.asarray(image, dtype=np.uint8)
    gray = rgb_to_gray(rgb)
    sobel_numba_cuda(gray)


def info():
    device = cuda.get_current_device()
    return {
        "Dispositivo CUDA": (
            device.name.decode() if isinstance(device.name, bytes) else device.name
        ),
        "Threads por bloque": f"{THREADS_PER_BLOCK}",
    }


def build_runner():
    return {
        "load": load_image,
        "gray": rgb_to_gray,
        "sobel": sobel_numba_cuda,
        "warmup": warmup,
        "info": info,
    }


if __name__ == "__main__":
    run_cli(build_runner(), "numba_cuda")
