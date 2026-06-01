import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = 5
DEFAULT_IMAGES = ("750x750", "1500x1500", "3000x3000", "6000x6000")
CPU_METHODS = (
    ("secuencial", "sobel_secuencial.py", None),
    ("numpy", "sobel_numpy.py", None),
    ("numba_parallel", "sobel_numba_parallel.py", None),
)
SCRIPT_FIELDNAMES = (
    "method",
    "device",
    "image",
    "runs",
    "transferencia_promedio_s",
    "rgb_gris_promedio_s",
    "sobel_promedio_s",
    "total_promedio_s",
    "blancos_pct",
)
FIELDNAMES = SCRIPT_FIELDNAMES + (
    "speed_up",
    "performance_pct",
)


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return tuple(devices)


def numba_cuda_available():
    try:
        from numba import cuda

        return cuda.is_available()
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta el benchmark Sobel y devuelve solo resultados CSV."
    )
    parser.add_argument(
        "--r",
        "--runs",
        dest="runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Cantidad de corridas por metodo y tamanio de imagen.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Cantidad maxima de scripts ejecutandose en paralelo. "
            "Usar mas de 1 acelera la recoleccion, pero contamina las mediciones."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Archivo CSV de salida. Si se omite, se imprime por stdout.",
    )
    parser.add_argument(
        "--images",
        default=",".join(DEFAULT_IMAGES),
        help="Resoluciones separadas por coma. Por defecto recorre todas las de la consigna.",
    )
    parser.add_argument(
        "--devices",
        default=",".join(available_devices()),
        help="Dispositivos PyTorch separados por coma: cpu, cuda o mps.",
    )
    return parser.parse_args()


def build_methods(devices):
    numba_cuda_methods = (
        (("numba_cuda", "sobel_numba_cuda.py", None),)
        if numba_cuda_available()
        else ()
    )
    pytorch_methods = tuple(
        (f"pytorch_{device}", "sobel_pytorch.py", device)
        for device in devices
    )
    return CPU_METHODS + numba_cuda_methods + pytorch_methods


def run_script(method, script_name, image, runs, device):
    command = [
        sys.executable,
        str(BASE_DIR / script_name),
        f"--r={runs}",
        f"--image={image}",
        "--csv",
    ]
    if device is not None:
        command.append(f"--device={device}")

    env = os.environ.copy()
    env.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))

    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Fallo {method} con imagen {image}.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    rows = list(csv.reader(completed.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(SCRIPT_FIELDNAMES):
        raise RuntimeError(
            f"Salida CSV inesperada para {method} {image}: {completed.stdout!r}"
        )

    return dict(zip(SCRIPT_FIELDNAMES, rows[0]))


def collect_results(runs, workers, images, devices):
    methods = build_methods(devices)
    tasks = {}
    max_workers = max(1, min(workers, len(methods) * len(images)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for method, script_name, device in methods:
            for image in images:
                future = executor.submit(
                    run_script,
                    method,
                    script_name,
                    image,
                    runs,
                    device,
                )
                tasks[future] = (method, image)

        results_by_key = {}
        for future in as_completed(tasks):
            method, image = tasks[future]
            results_by_key[(method, image)] = future.result()

    return [
        results_by_key[(method, image)]
        for method, _, _ in methods
        for image in images
    ]


def add_relative_metrics(rows):
    sequential_times = {
        row["image"]: float(row["total_promedio_s"])
        for row in rows
        if row["method"] == "secuencial"
    }

    for row in rows:
        sequential_time = sequential_times[row["image"]]
        method_time = float(row["total_promedio_s"])
        speed_up = sequential_time / method_time if method_time > 0 else 0.0
        row["speed_up"] = speed_up
        row["performance_pct"] = speed_up * 100

    return rows


def write_csv(rows, output):
    if output is None:
        stream = sys.stdout
        close_stream = False
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("w", newline="", encoding="utf-8")
        close_stream = True

    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close_stream:
            stream.close()


def split_values(raw_value, option):
    values = tuple(value.strip() for value in raw_value.split(",") if value.strip())
    if not values:
        raise ValueError(f"{option} debe incluir al menos un valor")
    return values


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--r/--runs debe ser mayor o igual a 1")
    if args.workers < 1:
        raise ValueError("--workers debe ser mayor o igual a 1")

    images = split_values(args.images, "--images")
    devices = split_values(args.devices, "--devices")
    invalid_devices = set(devices) - {"cpu", "cuda", "mps"}
    if invalid_devices:
        raise ValueError(f"Dispositivos invalidos: {', '.join(sorted(invalid_devices))}")

    rows = add_relative_metrics(
        collect_results(args.runs, args.workers, images, devices)
    )
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
