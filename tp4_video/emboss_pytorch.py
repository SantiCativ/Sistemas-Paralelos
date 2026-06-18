"""PyTorch implementation of the Emboss filter for CPU and CUDA."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from emboss_common import EMBOSS_KERNEL


def _prepare_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert an HxWx3 image to a float tensor in NCHW layout."""
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {image.shape}")
        tensor = torch.as_tensor(image)
    elif isinstance(image, torch.Tensor):
        tensor = image
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(image)!r}")

    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 tensor, got {tuple(tensor.shape)}")

    return tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)


def apply_kernel_torch(image: np.ndarray | torch.Tensor, kernel: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 kernel independently to each color channel with conv2d.

    The input and output layout is HxWx3. The device is inherited from the
    input tensor, so the same function works on CPU or CUDA.
    """
    tensor = _prepare_tensor(image)
    device = tensor.device
    dtype = tensor.dtype

    kernel_tensor = torch.as_tensor(kernel, dtype=dtype, device=device)
    if kernel_tensor.shape != (3, 3):
        raise ValueError(f"Expected 3x3 kernel, got {tuple(kernel_tensor.shape)}")

    weights = kernel_tensor.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    padded = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
    filtered = F.conv2d(padded, weights, groups=3)
    return filtered.squeeze(0).permute(1, 2, 0)


def emboss_torch(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Apply the Emboss filter with PyTorch and return uint8 HxWx3 output."""
    filtered = apply_kernel_torch(image, EMBOSS_KERNEL)
    return torch.clamp(filtered + 128.0, 0.0, 255.0).to(dtype=torch.uint8)
