import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = 5
DEFAULT_IMAGES = ("750x750", "1500x1500", "3000x3000", "6000x6000")
METHODS = (
    ("secuencial", "sobel_secuencial.py"),
    ("numpy", "sobel_numpy.py"),
    ("numba_parallel", "sobel_numba_parallel.py"),
)
FIELDNAMES = (
    "method",
    "image",
    "runs",
    "rgb_gris_promedio_s",
    "sobel_promedio_s",
    "total_promedio_s",
    "blancos_pct",
    "speed_up",
    "performance_pct",
)

SCRIPT_FIELDNAMES = FIELDNAMES[:7]


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
        default=os.cpu_count() or 1,
        help="Cantidad maxima de scripts ejecutandose en paralelo.",
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
    return parser.parse_args()


def run_script(method, script_name, image, runs):
    command = [
        sys.executable,
        str(BASE_DIR / script_name),
        f"--r={runs}",
        f"--image={image}",
        "--csv",
    ]
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


def collect_results(runs, workers, images):
    tasks = {}
    max_workers = max(1, min(workers, len(METHODS) * len(images)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for method, script_name in METHODS:
            for image in images:
                future = executor.submit(run_script, method, script_name, image, runs)
                tasks[future] = (method, image)

        results_by_key = {}
        for future in as_completed(tasks):
            method, image = tasks[future]
            results_by_key[(method, image)] = future.result()

    return [
        results_by_key[(method, image)]
        for method, _ in METHODS
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


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--r/--runs debe ser mayor o igual a 1")
    if args.workers < 1:
        raise ValueError("--workers debe ser mayor o igual a 1")

    images = tuple(image.strip() for image in args.images.split(",") if image.strip())
    if not images:
        raise ValueError("--images debe incluir al menos una resolucion")

    rows = add_relative_metrics(collect_results(args.runs, args.workers, images))
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
