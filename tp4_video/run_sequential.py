"""Run the sequential NumPy Emboss video pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from emboss_sequential import emboss_numpy
from video_processing_common import PipelineMetrics, elapsed_since, now, print_summary, process_video_frames


def run(
    input_path: str | Path,
    output_path: str | Path,
    fourcc: str = "mp4v",
    preserve_audio: bool = True,
    limit_frames: int | None = None,
) -> PipelineMetrics:
    """Process a video with the sequential NumPy implementation."""

    def processor(frame):
        start = now()
        processed = emboss_numpy(frame)
        return processed, {"filter": elapsed_since(start)}

    return process_video_frames(
        input_path=input_path,
        output_path=output_path,
        implementation="Secuencial",
        processor=processor,
        fourcc=fourcc,
        preserve_audio=preserve_audio,
        limit_frames=limit_frames,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="TP4 video Emboss secuencial con NumPy")
    parser.add_argument("--input", required=True, help="Ruta del video de entrada")
    parser.add_argument("--output", default="outputs/emboss_sequential.mp4", help="Ruta del video de salida")
    parser.add_argument("--codec", default="mp4v", help="FourCC para OpenCV VideoWriter")
    parser.add_argument("--no-audio", action="store_true", help="No intentar preservar audio con ffmpeg")
    parser.add_argument("--limit-frames", type=int, default=None, help="Procesar solo N frames")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    result = run(args.input, args.output, args.codec, not args.no_audio, args.limit_frames)
    print_summary([result])


if __name__ == "__main__":
    main()
