import argparse
import csv
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image


warnings.filterwarnings("ignore", category=DeprecationWarning)

IMAGE_DIR = Path(__file__).resolve().parent / "imagenes"
IMAGE_RESOLUTIONS = ("750x750", "1500x1500", "3000x3000", "6000x6000")


def parse_args(method_name):
    parser = argparse.ArgumentParser(description=f"Ejecuta Sobel usando {method_name}.")
    parser.add_argument(
        "--r",
        type=int,
        default=None,
        help="Cantidad de corridas. Si se omite, se hace una corrida visual.",
    )
    parser.add_argument(
        "--image",
        default="1500x1500",
        help="Resolucion a usar: 750x750, 1500x1500, 3000x3000 o 6000x6000.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Imprime solo una fila CSV con los promedios.",
    )
    return parser.parse_args()


def resolve_image_path(image):
    image = image.strip()
    candidates = [
        IMAGE_DIR / f"penguins_{image}.jpg",
        IMAGE_DIR / f"penguins_{image}.png",
        Path(image),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No se encontro la imagen '{image}'. Busque en: {expected}")


def timed_run(runner, image_path):
    load = runner.get("load")
    source = load(image_path) if load is not None else image_path

    gray_start = time.perf_counter()
    gray = runner["gray"](source)
    gray_seconds = time.perf_counter() - gray_start

    sobel_start = time.perf_counter()
    result = runner["sobel"](gray)
    sobel_seconds = time.perf_counter() - sobel_start
    total_seconds = gray_seconds + sobel_seconds

    return {
        "rgb_gris_s": gray_seconds,
        "sobel_s": sobel_seconds,
        "total_s": total_seconds,
        "blancos_pct": white_pixel_percent(result),
        "result": result,
    }


def white_pixel_percent(result):
    array = np.asarray(result, dtype=np.float32)
    white_pixels = np.count_nonzero(array >= 255)
    return white_pixels * 100 / array.size


def run_benchmark(runner, method_name, image_name, runs):
    if runs < 1:
        raise ValueError("--r debe ser mayor o igual a 1")

    image_path = resolve_image_path(image_name)
    rgb_gray_times = []
    sobel_times = []
    total_times = []
    white_pixel_percentages = []
    last_result = None

    warmup = runner.get("warmup")
    if warmup is not None:
        warmup(image_path)

    for _ in range(runs):
        timing = timed_run(runner, image_path)
        rgb_gray_times.append(timing["rgb_gris_s"])
        sobel_times.append(timing["sobel_s"])
        total_times.append(timing["total_s"])
        white_pixel_percentages.append(timing["blancos_pct"])
        last_result = timing["result"]

    return {
        "method": method_name,
        "image": image_name,
        "runs": runs,
        "rgb_gris_promedio_s": statistics.fmean(rgb_gray_times),
        "sobel_promedio_s": statistics.fmean(sobel_times),
        "total_promedio_s": statistics.fmean(total_times),
        "blancos_pct": statistics.fmean(white_pixel_percentages),
        "last_result": last_result,
    }


def print_human_summary(summary):
    print(f"Metodo: {summary['method']}")
    print(f"Imagen: {summary['image']}")
    print(f"Corridas: {summary['runs']}")
    for label, value in summary.get("info", {}).items():
        print(f"{label}: {value}")
    print(f"RGB->gris promedio: {summary['rgb_gris_promedio_s']:.6f} s")
    print(f"Sobel promedio: {summary['sobel_promedio_s']:.6f} s")
    print(f"Total promedio: {summary['total_promedio_s']:.6f} s")
    print(f"Pixeles blancos: {summary['blancos_pct']:.6f} %")


def print_csv_summary(summary, include_header=False):
    fieldnames = [
        "method",
        "image",
        "runs",
        "rgb_gris_promedio_s",
        "sobel_promedio_s",
        "total_promedio_s",
        "blancos_pct",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    if include_header:
        writer.writeheader()
    writer.writerow({field: summary[field] for field in fieldnames})


def show_result(result):
    array = np.asarray(result, dtype=np.float32)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")
    image.show()


def run_cli(runner, method_name):
    args = parse_args(method_name)
    runs = args.r if args.r is not None else 1
    summary = run_benchmark(runner, method_name, args.image, runs)
    info = runner.get("info")
    if info is not None:
        summary["info"] = info()

    if args.csv:
        print_csv_summary(summary)
    else:
        print_human_summary(summary)

    if args.r is None and not args.csv:
        show_result(summary["last_result"])
