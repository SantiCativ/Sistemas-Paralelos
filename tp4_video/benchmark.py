"""Benchmark runner for all TP4 Emboss video implementations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import run_pytorch_cpu
import run_pytorch_cuda
import run_sequential
from video_processing_common import (
    PipelineMetrics,
    print_summary,
    write_markdown_summary,
    write_results_csv,
)


def _output_path(output_dir: Path, suffix: str) -> Path:
    """Build one output video path inside the benchmark output directory."""
    return output_dir / f"emboss_{suffix}.mp4"


def calculate_speedups(results: list[PipelineMetrics]) -> None:
    """Fill speedup values in-place using the sequential total as baseline."""
    baseline = next((item for item in results if item.implementacion == "Secuencial"), None)
    if baseline is None or baseline.tiempo_total <= 0:
        return

    for item in results:
        item.speedup = baseline.tiempo_total / item.tiempo_total if item.tiempo_total > 0 else 0.0


def run_benchmark(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    csv_path: str | Path = "outputs/resultados_tp4.csv",
    markdown_path: str | Path = "outputs/resumen_tp4.md",
    fourcc: str = "mp4v",
    preserve_audio: bool = True,
    limit_frames: int | None = None,
    skip_cuda: bool = False,
) -> list[PipelineMetrics]:
    """Run all requested implementations and export summaries."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results: list[PipelineMetrics] = []
    results.append(run_sequential.run(input_path, _output_path(output, "sequential"), fourcc, preserve_audio, limit_frames))
    results.append(run_pytorch_cpu.run(input_path, _output_path(output, "pytorch_cpu"), fourcc, preserve_audio, limit_frames))

    if not skip_cuda and torch.cuda.is_available():
        results.append(run_pytorch_cuda.run(input_path, _output_path(output, "pytorch_cuda"), fourcc, preserve_audio, limit_frames))
    elif not skip_cuda:
        print("CUDA no disponible: se omite PyTorch CUDA.")

    calculate_speedups(results)
    write_results_csv(results, csv_path)
    write_markdown_summary(results, markdown_path)
    print_summary(results)
    print(f"CSV generado: {csv_path}")
    print(f"Markdown generado: {markdown_path}")
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark TP4 video Emboss")
    parser.add_argument("--input", required=True, help="Ruta del video 4K de entrada")
    parser.add_argument("--output-dir", default="outputs", help="Directorio para videos procesados")
    parser.add_argument("--csv", default="outputs/resultados_tp4.csv", help="CSV de metricas")
    parser.add_argument("--markdown", default="outputs/resumen_tp4.md", help="Resumen Markdown")
    parser.add_argument("--codec", default="mp4v", help="FourCC para OpenCV VideoWriter")
    parser.add_argument("--no-audio", action="store_true", help="No intentar preservar audio con ffmpeg")
    parser.add_argument("--limit-frames", type=int, default=None, help="Procesar solo N frames")
    parser.add_argument("--skip-cuda", action="store_true", help="Omitir PyTorch CUDA")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_benchmark(
        input_path=args.input,
        output_dir=args.output_dir,
        csv_path=args.csv,
        markdown_path=args.markdown,
        fourcc=args.codec,
        preserve_audio=not args.no_audio,
        limit_frames=args.limit_frames,
        skip_cuda=args.skip_cuda,
    )


if __name__ == "__main__":
    main()
