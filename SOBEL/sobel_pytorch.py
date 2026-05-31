import numpy as np
import torch
from PIL import Image

from sobel_common import run_cli


def autodetect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero torch.cuda.is_available() es False")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Se solicito MPS, pero torch.backends.mps.is_available() es False")


def gray_torch(image):
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return torch.clamp(gray, 0, 255)


def sobel_torch(image):
    result = torch.zeros_like(image)

    top_left = image[:-2, :-2]
    top = image[:-2, 1:-1]
    top_right = image[:-2, 2:]
    left = image[1:-1, :-2]
    right = image[1:-1, 2:]
    bottom_left = image[2:, :-2]
    bottom = image[2:, 1:-1]
    bottom_right = image[2:, 2:]

    gx = -top_left + top_right - 2.0 * left + 2.0 * right - bottom_left + bottom_right
    gy = -top_left - 2.0 * top - top_right + bottom_left + 2.0 * bottom + bottom_right
    result[1:-1, 1:-1] = torch.abs(gx) + torch.abs(gy)

    return result


def build_runner():
    state = {"device": autodetect_device()}

    def configure(args):
        state["device"] = args.device or autodetect_device()
        validate_device(state["device"])

    def load_step(image_path):
        rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(rgb)

    def transfer_in(rgb):
        return rgb.to(state["device"])

    def transfer_out(result):
        return result.detach().cpu().numpy()

    def synchronize():
        if state["device"] == "cuda":
            torch.cuda.synchronize()
        elif state["device"] == "mps":
            torch.mps.synchronize()

    def warmup(image_path):
        image = Image.open(image_path).convert("RGB").resize((8, 8))
        rgb = torch.from_numpy(np.array(image, dtype=np.float32))
        if state["device"] != "cpu":
            rgb = transfer_in(rgb)
        result = sobel_torch(gray_torch(rgb))
        if state["device"] != "cpu":
            transfer_out(result)
        synchronize()

    def info():
        return {
            "CUDA disponible": torch.cuda.is_available(),
            "MPS disponible": torch.backends.mps.is_available(),
        }

    return {
        "load": load_step,
        "transfer_in": transfer_in,
        "gray": gray_torch,
        "sobel": sobel_torch,
        "transfer_out": transfer_out,
        "synchronize": synchronize,
        "warmup": warmup,
        "configure": configure,
        "device": lambda: state["device"],
        "measure_transfer": lambda: state["device"] != "cpu",
        "info": info,
    }


if __name__ == "__main__":
    run_cli(build_runner(), "pytorch")
