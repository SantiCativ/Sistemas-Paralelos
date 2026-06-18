"""Sequential CPU implementation of the Emboss filter using NumPy."""

from __future__ import annotations

import numpy as np

from emboss_common import EMBOSS_KERNEL, clamp_image, validate_image, validate_kernel


# La función aplica una convolución 3×3 utilizando el kernel Emboss sobre una imagen RGB.
def apply_kernel_numpy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    # Verifica que la imagen tenga formato correcto (H×W×3)
    validate_image(image)
    # Verifica que el kernel sea 3×3
    validate_kernel(kernel)
    # Convierte los píxeles a float32 para evitar problemas durante los cálculos.
    source = image.astype(np.float32, copy=False)
    # Esto agrega un borde de 1 píxel alrededor de toda la imagen. El objetivo Poder calcular la convolución en los bordes de la imagen.
    padded = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result = np.zeros_like(source, dtype=np.float32)

    for row in range(3):
        for col in range(3):
            result += (
                kernel[row, col]
                * padded[row : row + image.shape[0], col : col + image.shape[1], :]
            )

    return result


def emboss_numpy(image: np.ndarray) -> np.ndarray:
    """Apply the Emboss filter on CPU using only NumPy operations."""
    filtered = apply_kernel_numpy(image, EMBOSS_KERNEL)
    return clamp_image(filtered + 128.0)
