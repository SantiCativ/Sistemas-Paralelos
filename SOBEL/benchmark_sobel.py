import argparse
import ast
import csv
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SIZES = (750, 1500, 3000, 6000)
METHODS = ("sequential", "numpy", "numba")
METHOD_LABELS = {
    "sequential": "secuencial",
    "numpy": "numpy",
    "numba": "numba paralelo CPU",
}
FUNCTION_CACHE = {}


@dataclass
class RunResult:
    size: int
    method: str
    gray_time: float
    sobel_time: float
    total_time: float
    white_pct: float


@dataclass
class BenchmarkResult:
    size: int
    method: str
    gray_avg: float
    sobel_avg: float
    total_avg: float
    white_pct: float


def load_function_defs(script_path, globals_dict):
    """Carga solo las funciones del script, evitando ejecutar su codigo principal."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    function_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(script_path), "exec"), globals_dict)
    return globals_dict


def load_sequential_functions(base_dir):
    if "sequential" in FUNCTION_CACHE:
        return FUNCTION_CACHE["sequential"]

    namespace = {}
    load_function_defs(base_dir / "sobel_sequential.py", namespace)
    FUNCTION_CACHE["sequential"] = namespace["rgb_to_gray"], namespace["sobel_sequential"]
    return FUNCTION_CACHE["sequential"]


def load_numba_functions(base_dir):
    if "numba" in FUNCTION_CACHE:
        return FUNCTION_CACHE["numba"]

    try:
        from numba import njit, prange, set_num_threads
    except ImportError:
        return None

    try:
        set_num_threads(os.cpu_count() or 1)
    except ValueError:
        pass

    namespace = {"np": np, "njit": njit, "prange": prange}
    load_function_defs(base_dir / "sobel_numba_parallel.py", namespace)
    FUNCTION_CACHE["numba"] = namespace["rgb_to_gray"], namespace["sobel_parallel"]
    return FUNCTION_CACHE["numba"]


def image_path_for_size(image_dir, size):
    exact = image_dir / f"penguins_{size}x{size}.jpg"
    if exact.exists():
        return exact

    plain = image_dir / "penguins.jpg"
    if size == 1500:
        return None

    return plain if plain.exists() else None


def load_rgb_numpy(image_dir, size):
    image_path = image_path_for_size(image_dir, size)

    if image_path is None:
        source_path = image_dir / "penguins_3000x3000.jpg"
        image = Image.open(source_path).convert("RGB")
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    else:
        image = Image.open(image_path).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)

    return np.asarray(image, dtype=np.float32)


def to_sequential_rgb(rgb_np):
    height, width, _ = rgb_np.shape
    pixels = [tuple(map(int, pixel)) for pixel in rgb_np.reshape(height * width, 3)]
    return [pixels[i * width : (i + 1) * width] for i in range(height)]


def rgb_to_gray_numpy(rgb):
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def sobel_numpy_like_script(gray):
    img = gray.astype(np.float32)
    height, width = img.shape

    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    result = np.zeros((height, width), dtype=np.float32)

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            p = img[row - 1 : row + 2, col - 1 : col + 2]
            gx = np.sum(p * gx_kernel)
            gy = np.sum(p * gy_kernel)
            result[row, col] = np.sqrt(gx * gx + gy * gy)

    return result


def output_as_u8(result):
    return np.clip(np.asarray(result), 0, 255).astype(np.uint8)


def white_percentage(result):
    result_u8 = output_as_u8(result)
    return float(np.count_nonzero(result_u8 == 255) * 100 / result_u8.size)


def save_result_image(result, output_path):
    Image.fromarray(output_as_u8(result), mode="L").save(output_path)


def build_method(base_dir, method, rgb_np):
    height, width, _ = rgb_np.shape

    if method == "sequential":
        rgb_to_gray, sobel_sequential = load_sequential_functions(base_dir)
        return (
            rgb_to_gray,
            lambda gray: sobel_sequential(gray, width, height),
            to_sequential_rgb(rgb_np),
        )

    if method == "numpy":
        return rgb_to_gray_numpy, sobel_numpy_like_script, rgb_np

    if method == "numba":
        numba_functions = load_numba_functions(base_dir)
        if numba_functions is None:
            return None

        rgb_to_gray, sobel_parallel = numba_functions
        return rgb_to_gray, sobel_parallel, rgb_np

    raise ValueError(f"Metodo desconocido: {method}")


def warmup_method(base_dir, method):
    if method != "numba":
        return True

    sample = np.zeros((32, 32, 3), dtype=np.float32)
    method_data = build_method(base_dir, method, sample)
    if method_data is None:
        return False

    gray_func, sobel_func, rgb_input = method_data
    gray = gray_func(rgb_input)
    sobel_func(gray)
    return True


def run_once(base_dir, image_dir, method, size, save_path=None):
    rgb_np = load_rgb_numpy(image_dir, size)
    method_data = build_method(base_dir, method, rgb_np)

    if method_data is None:
        raise RuntimeError("Numba no esta instalado en el entorno activo.")

    gray_func, sobel_func, rgb_input = method_data

    total_start = time.perf_counter()

    gray_start = time.perf_counter()
    gray = gray_func(rgb_input)
    gray_end = time.perf_counter()

    sobel_start = time.perf_counter()
    result = sobel_func(gray)
    sobel_end = time.perf_counter()

    if save_path is not None:
        save_result_image(result, save_path)

    return RunResult(
        size=size,
        method=METHOD_LABELS[method],
        gray_time=gray_end - gray_start,
        sobel_time=sobel_end - sobel_start,
        total_time=sobel_end - total_start,
        white_pct=white_percentage(result),
    )


def run_benchmark(base_dir, image_dir, methods, sizes, runs):
    results = []

    for method in methods:
        if method == "numba" and not warmup_method(base_dir, method):
            print("Numba no esta instalado: se omite 'numba paralelo CPU'.")

    for size in sizes:
        for method in methods:
            if method == "numba" and load_numba_functions(base_dir) is None:
                continue

            print(f"Ejecutando {METHOD_LABELS[method]} - {size}x{size} ({runs} corridas)...")
            run_results = [
                run_once(base_dir, image_dir, method, size)
                for _ in range(runs)
            ]

            results.append(
                BenchmarkResult(
                    size=size,
                    method=METHOD_LABELS[method],
                    gray_avg=statistics.fmean(r.gray_time for r in run_results),
                    sobel_avg=statistics.fmean(r.sobel_time for r in run_results),
                    total_avg=statistics.fmean(r.total_time for r in run_results),
                    white_pct=statistics.fmean(r.white_pct for r in run_results),
                )
            )

    return results


def hardware_info():
    info = {
        "Sistema operativo": f"{platform.system()} {platform.release()}",
        "Version de Python": platform.python_version(),
        "CPU": platform.processor() or platform.machine(),
        "Nucleos logicos": str(os.cpu_count() or "desconocido"),
    }

    try:
        import psutil

        info["Nucleos fisicos"] = str(psutil.cpu_count(logical=False) or "desconocido")
        info["RAM disponible"] = f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB"
    except ImportError:
        info["Nucleos fisicos"] = "psutil no instalado"
        info["RAM disponible"] = "psutil no instalado"

    return info


def grouped_by_size(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result.size, []).append(result)
    return grouped


def format_markdown(results, runs):
    lines = ["# Benchmark Sobel", "", "## Entorno y hardware", ""]

    for key, value in hardware_info().items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            f"- Corridas por metodo y tamano: {runs}",
            "",
            "Los tiempos excluyen carga y guardado de imagenes. El porcentaje de blancos se calcula sobre la salida Sobel recortada a uint8.",
            "",
            "## Resultados",
            "",
        ]
    )

    for size, size_results in grouped_by_size(results).items():
        sequential = next((r for r in size_results if r.method == "secuencial"), None)
        baseline = sequential.total_avg if sequential else None

        lines.extend(
            [
                f"### {size}x{size}",
                "",
                "| metodo | tiempo RGB->gris (s) | tiempo Sobel (s) | tiempo total (s) | % blancos | speed-up | performance (%) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for result in size_results:
            speedup = baseline / result.total_avg if baseline else 1.0
            performance = speedup * 100
            lines.append(
                f"| {result.method} | {result.gray_avg:.6f} | {result.sobel_avg:.6f} | "
                f"{result.total_avg:.6f} | {result.white_pct:.4f} | {speedup:.4f} | {performance:.2f} |"
            )

        lines.append("")

    return "\n".join(lines)


def write_csv(results, output_path):
    csv_path = output_path.with_suffix(".csv")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tamano",
                "metodo",
                "tiempo_rgb_gris_s",
                "tiempo_sobel_s",
                "tiempo_total_s",
                "porcentaje_blancos",
                "speed_up",
                "performance_pct",
            ]
        )

        for size, size_results in grouped_by_size(results).items():
            sequential = next((r for r in size_results if r.method == "secuencial"), None)
            baseline = sequential.total_avg if sequential else None

            for result in size_results:
                speedup = baseline / result.total_avg if baseline else 1.0
                writer.writerow(
                    [
                        f"{size}x{size}",
                        result.method,
                        f"{result.gray_avg:.6f}",
                        f"{result.sobel_avg:.6f}",
                        f"{result.total_avg:.6f}",
                        f"{result.white_pct:.4f}",
                        f"{speedup:.4f}",
                        f"{speedup * 100:.2f}",
                    ]
                )

    return csv_path


def parse_methods(values):
    methods = tuple(values)
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Metodos invalidos: {', '.join(invalid)}")
    return methods


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runner externo para medir las implementaciones Sobel sin modificar scripts."
    )
    subparsers = parser.add_subparsers(dest="command")

    benchmark = subparsers.add_parser("benchmark", help="Ejecuta la consigna completa.")
    benchmark.add_argument("--runs", type=int, default=5)
    benchmark.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    benchmark.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    benchmark.add_argument("--image-dir", type=Path, default=Path("imagenes"))
    benchmark.add_argument("--output", type=Path, default=Path("resultados_benchmark.md"))

    run = subparsers.add_parser("run", help="Ejecuta un metodo para un tamano puntual.")
    run.add_argument("--method", required=True, choices=METHODS)
    run.add_argument("--size", type=int, required=True)
    run.add_argument("--image-dir", type=Path, default=Path("imagenes"))
    run.add_argument("--save", type=Path, help="Guarda la imagen Sobel resultante.")

    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.command is None:
        args.command = "benchmark"
        args.runs = 5
        args.sizes = DEFAULT_SIZES
        args.methods = METHODS
        args.image_dir = Path("imagenes")
        args.output = Path("resultados_benchmark.md")

    image_dir = args.image_dir
    if not image_dir.is_absolute():
        image_dir = base_dir / image_dir

    if args.command == "run":
        save_path = args.save
        if save_path is not None and not save_path.is_absolute():
            save_path = base_dir / save_path

        if args.method == "numba":
            warmup_method(base_dir, args.method)

        result = run_once(base_dir, image_dir, args.method, args.size, save_path)
        print(json.dumps(result.__dict__, indent=2))
        return

    if args.runs < 1:
        raise ValueError("--runs debe ser al menos 1")

    results = run_benchmark(base_dir, image_dir, parse_methods(args.methods), args.sizes, args.runs)
    report = format_markdown(results, args.runs)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    output_path.write_text(report, encoding="utf-8")
    csv_path = write_csv(results, output_path)

    print(f"Reporte Markdown: {output_path}")
    print(f"Resultados CSV: {csv_path}")


if __name__ == "__main__":
    main()
