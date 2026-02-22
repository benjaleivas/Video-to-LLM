"""Zero-config entry point: videopipe <video> <output_dir>

Probes the video, auto-tunes parameters, and runs the full pipeline.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from .cli import _pipeline
from .probe import probe_video
from .tuner import tune_parameters
from .utils import log


def main(argv: list[str] | None = None) -> int:
    args_list = argv if argv is not None else sys.argv[1:]

    if len(args_list) != 2 or args_list[0].startswith("-"):
        print("Usage: videopipe <video_path> <output_dir>")
        print()
        print("  Analyzes the video, auto-tunes all parameters, and runs the pipeline.")
        print()
        print("  For advanced usage with manual parameter control:")
        print("    python -m videopipe --video <path> --out <dir> [options]")
        return 1 if args_list else 0

    video_path = Path(args_list[0]).expanduser().resolve()
    output_dir = Path(args_list[1]).expanduser().resolve()

    if not video_path.exists():
        log(f"ERROR: video not found: {video_path}")
        return 1
    if not video_path.is_file():
        log(f"ERROR: not a file: {video_path}")
        return 1

    log(f"Video: {video_path}")
    log(f"Output: {output_dir}")
    log("")

    start = time.time()

    # Pass 1: probe
    log("=== Pass 1/2: Probing video ===")
    probe_result = probe_video(video_path)
    probe_elapsed = time.time() - start
    log(f"Probe completed in {probe_elapsed:.1f}s")
    log("")

    # Tune parameters from probe
    log("=== Auto-tuning parameters ===")
    pipeline_args = tune_parameters(
        probe_result,
        video_path=str(video_path),
        output_dir=str(output_dir),
    )

    if not probe_result.has_audio:
        log(
            "Warning: no audio stream detected — transcription will be skipped or empty"
        )
    log("")

    # Pass 2: full pipeline
    log("=== Pass 2/2: Running pipeline ===")
    try:
        result = _pipeline(pipeline_args)
    except Exception as exc:
        log(f"ERROR: {exc}")
        if os.getenv("VIDEOPIPE_DEBUG"):
            traceback.print_exc()
        else:
            log("  Set VIDEOPIPE_DEBUG=1 for full traceback")
        return 1

    total_elapsed = time.time() - start
    log(
        f"Total time: {total_elapsed:.1f}s (probe: {probe_elapsed:.1f}s, pipeline: {total_elapsed - probe_elapsed:.1f}s)"
    )
    return result
