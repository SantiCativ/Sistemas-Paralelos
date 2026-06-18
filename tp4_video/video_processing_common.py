"""Reusable video processing, timing, audio and metrics utilities."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from emboss_common import ensure_parent_dir, format_seconds, metric_float

FrameProcessor = Callable[[np.ndarray], tuple[np.ndarray, dict[str, float]]]


@dataclass(slots=True)
class VideoMetadata:
    """Metadata obtained from OpenCV VideoCapture."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def resolution(self) -> str:
        """Return resolution as WIDTHxHEIGHT."""
        return f"{self.width}x{self.height}"


@dataclass(slots=True)
class PipelineMetrics:
    """Metrics collected for one implementation."""

    implementacion: str
    frames: int
    width: int
    height: int
    fps_original: float
    fps_efectivos: float
    tiempo_lectura: float
    tiempo_filtrado: float
    tiempo_escritura: float
    tiempo_cpu_gpu: float
    tiempo_gpu_cpu: float
    tiempo_total: float
    ram_mb: float
    gpu_mb: float
    codec: str
    speedup: float = 1.0
    ffmpeg_usado: bool = False
    output_path: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    def csv_row(self) -> dict[str, float | int | str]:
        """Return the mandatory CSV columns requested by the assignment."""
        return {
            "implementacion": self.implementacion,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "fps_original": self.fps_original,
            "fps_efectivos": self.fps_efectivos,
            "tiempo_lectura": self.tiempo_lectura,
            "tiempo_filtrado": self.tiempo_filtrado,
            "tiempo_escritura": self.tiempo_escritura,
            "tiempo_cpu_gpu": self.tiempo_cpu_gpu,
            "tiempo_gpu_cpu": self.tiempo_gpu_cpu,
            "tiempo_total": self.tiempo_total,
            "ram_mb": self.ram_mb,
            "gpu_mb": self.gpu_mb,
            "codec": self.codec,
            "speedup": self.speedup,
        }


CSV_COLUMNS = [
    "implementacion",
    "frames",
    "width",
    "height",
    "fps_original",
    "fps_efectivos",
    "tiempo_lectura",
    "tiempo_filtrado",
    "tiempo_escritura",
    "tiempo_cpu_gpu",
    "tiempo_gpu_cpu",
    "tiempo_total",
    "ram_mb",
    "gpu_mb",
    "codec",
    "speedup",
]


def now() -> float:
    """Return a monotonic high-resolution timestamp."""
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    """Return elapsed seconds since a timestamp."""
    return now() - start


def open_video(input_path: str | Path) -> cv2.VideoCapture:
    """Open a video file with OpenCV and fail with a clear error if needed."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    return capture


def get_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read width, height, FPS and estimated frame count from a capture."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count)


def create_video_writer(output_path: str | Path, metadata: VideoMetadata, fourcc: str = "mp4v") -> cv2.VideoWriter:
    """Create a VideoWriter with the original resolution and FPS."""
    if len(fourcc) != 4:
        raise ValueError("fourcc must contain exactly 4 characters")

    output = ensure_parent_dir(output_path)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*fourcc),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create VideoWriter for {output}")
    return writer


def get_ram_mb() -> float:
    """Return current process RSS memory in MiB."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except Exception:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage / 1024.0)


def get_gpu_mb() -> float:
    """Return peak CUDA allocated memory in MiB, or 0 when unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        return 0.0
    return 0.0


def _run_ffmpeg(command: list[str]) -> bool:
    """Run ffmpeg silently and return True when it succeeds."""
    if shutil.which(command[0]) is None:
        return False
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return completed.returncode == 0


def extract_audio(input_path: str | Path, audio_path: str | Path) -> bool:
    """Extract the first audio stream from a video using ffmpeg."""
    return _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "copy",
            str(audio_path),
        ]
    )


def mux_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> bool:
    """Merge a processed silent video with an audio stream using ffmpeg."""
    return _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    )


def process_video_frames(
    input_path: str | Path,
    output_path: str | Path,
    implementation: str,
    processor: FrameProcessor,
    fourcc: str = "mp4v",
    preserve_audio: bool = True,
    limit_frames: int | None = None,
) -> PipelineMetrics:
    """Process a video frame by frame and collect mandatory metrics."""
    input_file = Path(input_path)
    final_output = ensure_parent_dir(output_path)
    use_temp_video = preserve_audio
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    encoded_output = final_output
    audio_extracted = False
    ffmpeg_used = False

    if use_temp_video:
        temp_dir = tempfile.TemporaryDirectory(prefix="tp4_video_")
        encoded_output = Path(temp_dir.name) / f"{final_output.stem}_silent{final_output.suffix}"
        audio_extracted = extract_audio(input_file, Path(temp_dir.name) / "audio.aac")

    capture = open_video(input_file)
    metadata = get_video_metadata(capture)
    writer = create_video_writer(encoded_output, metadata, fourcc)

    frames = 0
    read_time = 0.0
    filter_time = 0.0
    write_time = 0.0
    cpu_gpu_time = 0.0
    gpu_cpu_time = 0.0
    max_ram = get_ram_mb()
    total_start = now()

    try:
        while True:
            if limit_frames is not None and frames >= limit_frames:
                break

            start = now()
            ok, frame = capture.read()
            read_time += elapsed_since(start)

            if not ok:
                break

            processed, timing = processor(frame)
            filter_time += timing.get("filter", 0.0)
            cpu_gpu_time += timing.get("cpu_gpu", 0.0)
            gpu_cpu_time += timing.get("gpu_cpu", 0.0)

            start = now()
            writer.write(processed)
            write_time += elapsed_since(start)

            frames += 1
            current_ram = get_ram_mb()
            if current_ram > max_ram:
                max_ram = current_ram
    finally:
        writer.release()
        capture.release()

    total_time = elapsed_since(total_start)

    if preserve_audio and temp_dir is not None and audio_extracted:
        audio_path = Path(temp_dir.name) / "audio.aac"
        ffmpeg_used = mux_audio(encoded_output, audio_path, final_output)
        if not ffmpeg_used:
            shutil.copyfile(encoded_output, final_output)
    elif preserve_audio and encoded_output != final_output:
        shutil.copyfile(encoded_output, final_output)

    if temp_dir is not None:
        temp_dir.cleanup()

    fps_effective = frames / total_time if total_time > 0 else 0.0
    codec_label = fourcc if not ffmpeg_used else f"{fourcc} + ffmpeg audio"

    return PipelineMetrics(
        implementacion=implementation,
        frames=frames,
        width=metadata.width,
        height=metadata.height,
        fps_original=metadata.fps,
        fps_efectivos=fps_effective,
        tiempo_lectura=read_time,
        tiempo_filtrado=filter_time,
        tiempo_escritura=write_time,
        tiempo_cpu_gpu=cpu_gpu_time,
        tiempo_gpu_cpu=gpu_cpu_time,
        tiempo_total=total_time,
        ram_mb=max_ram,
        gpu_mb=get_gpu_mb(),
        codec=codec_label,
        ffmpeg_usado=ffmpeg_used,
        output_path=str(final_output),
    )


def write_results_csv(results: Iterable[PipelineMetrics], output_path: str | Path) -> None:
    """Write mandatory benchmark metrics to CSV."""
    output = ensure_parent_dir(output_path)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.csv_row())


def build_markdown_summary(results: Iterable[PipelineMetrics]) -> str:
    """Build a Markdown table suitable for the practical report."""
    rows = list(results)
    lines = [
        "# Resumen TP4 - Video Emboss",
        "",
        "| Implementacion | Frames | Resolucion | FPS original | FPS efectivos | Lectura (s) | Filtrado (s) | Escritura (s) | CPU->GPU (s) | GPU->CPU (s) | Total (s) | RAM MB | GPU MB | Codec | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in rows:
        lines.append(
            "| "
            f"{item.implementacion} | {item.frames} | {item.width}x{item.height} | "
            f"{item.fps_original:.3f} | {item.fps_efectivos:.3f} | "
            f"{format_seconds(item.tiempo_lectura)} | {format_seconds(item.tiempo_filtrado)} | "
            f"{format_seconds(item.tiempo_escritura)} | {format_seconds(item.tiempo_cpu_gpu)} | "
            f"{format_seconds(item.tiempo_gpu_cpu)} | {format_seconds(item.tiempo_total)} | "
            f"{item.ram_mb:.2f} | {item.gpu_mb:.2f} | {item.codec} | {item.speedup:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_markdown_summary(results: Iterable[PipelineMetrics], output_path: str | Path) -> None:
    """Write a Markdown summary file."""
    output = ensure_parent_dir(output_path)
    output.write_text(build_markdown_summary(results), encoding="utf-8")


def print_summary(results: Iterable[PipelineMetrics]) -> None:
    """Print a compact human-readable benchmark summary."""
    for result in results:
        print(
            f"{result.implementacion:14s} "
            f"frames={result.frames} "
            f"res={result.width}x{result.height} "
            f"fps_eff={result.fps_efectivos:.2f} "
            f"read={result.tiempo_lectura:.3f}s "
            f"filter={result.tiempo_filtrado:.3f}s "
            f"write={result.tiempo_escritura:.3f}s "
            f"cpu_gpu={result.tiempo_cpu_gpu:.3f}s "
            f"gpu_cpu={result.tiempo_gpu_cpu:.3f}s "
            f"total={result.tiempo_total:.3f}s "
            f"ram={result.ram_mb:.1f}MB "
            f"gpu={result.gpu_mb:.1f}MB "
            f"speedup={result.speedup:.4f}"
        )


def metrics_to_dict(result: PipelineMetrics) -> dict[str, object]:
    """Convert a metrics dataclass to a plain dictionary."""
    data = asdict(result)
    data["csv"] = result.csv_row()
    data["tiempo_total"] = metric_float(data["tiempo_total"])
    return data
