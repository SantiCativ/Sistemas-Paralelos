"""Shared definitions and helpers for the Emboss filter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EMBOSS_KERNEL: np.ndarray = np.array(
    [
        [-2.0, -1.0, 0.0],
        [-1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
    ],
    dtype=np.float32,
)


def validate_image(image: np.ndarray) -> None:
    """Validate that an image has three color channels in HxWx3 layout."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)!r}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with shape HxWx3, got {image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")


def validate_kernel(kernel: np.ndarray) -> None:
    """Validate that a kernel is a 3x3 numeric matrix."""
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray kernel, got {type(kernel)!r}")

    if kernel.shape != (3, 3):
        raise ValueError(f"Expected 3x3 kernel, got {kernel.shape}")


def clamp_image(image: np.ndarray) -> np.ndarray:
    """Clamp a numeric image to uint8 values in the [0, 255] range."""
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a path and return it as Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def format_seconds(value: float | None) -> str:
    """Format optional seconds for console and Markdown summaries."""
    if value is None:
        return "-"
    return f"{value:.6f}"


def metric_float(value: Any) -> float:
    """Normalize numeric values for CSV output."""
    if value is None:
        return 0.0
    return float(value)
