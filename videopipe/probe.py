"""Video probe: analyze a video's characteristics before the full pipeline run.

Extracts metadata, samples frames, measures scene change frequency,
runs lightweight OCR on a handful of frames, and checks for audio.
The result feeds into tuner.py to auto-select optimal pipeline parameters.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg_utils import extract_sample_frames, extract_scene_frames
from .utils import (
    check_binaries,
    ffprobe_duration_seconds,
    format_duration,
    log,
    log_verbose,
    run_cmd,
)


@dataclass
class ProbeResult:
    duration_seconds: float
    width: int
    height: int
    fps: float
    scene_changes_per_minute: float
    avg_text_length: float
    avg_ocr_confidence: float
    has_audio: bool
    sample_frame_count: int
    scene_frame_count: int


def ffprobe_video_info(video_path: Path) -> dict:
    """Return width, height, fps, and whether an audio stream exists."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0:s=,",
        str(video_path),
    ]
    result = run_cmd(cmd, capture_output=True, check=False)
    parts = (result.stdout or "").strip().split(",")

    width, height, fps = 1920, 1080, 30.0
    if len(parts) >= 3:
        try:
            width = int(parts[0])
        except ValueError:
            pass
        try:
            height = int(parts[1])
        except ValueError:
            pass
        try:
            fps_str = parts[2]
            if "/" in fps_str:
                num, den = fps_str.split("/", 1)
                fps = float(num) / float(den) if float(den) else 30.0
            else:
                fps = float(fps_str)
        except (ValueError, ZeroDivisionError):
            pass
    fps = max(0.1, min(fps, 240.0))

    # Check for audio stream
    audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    audio_result = run_cmd(audio_cmd, capture_output=True, check=False)
    has_audio = bool((audio_result.stdout or "").strip())

    return {"width": width, "height": height, "fps": fps, "has_audio": has_audio}


def _probe_text_density(
    sample_frames: list[dict], max_frames: int = 5
) -> tuple[float, float]:
    """Run lightweight Tesseract OCR on a few sample frames.

    Returns (avg_text_length, avg_confidence). Falls back to conservative
    defaults if Tesseract is not installed.
    """
    if not sample_frames:
        return 200.0, 70.0

    if shutil.which("tesseract") is None:
        log("Probe: Tesseract not installed, skipping text density analysis")
        return 200.0, 70.0

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        log("Probe: pytesseract/Pillow not available, skipping text density analysis")
        return 200.0, 70.0

    # Pick evenly-spaced frames from the samples
    step = max(1, len(sample_frames) // max_frames)
    chosen = sample_frames[::step][:max_frames]

    total_len = 0.0
    total_conf = 0.0
    count = 0

    for entry in chosen:
        raw_path = entry.get("raw_path", "")
        if not raw_path or not Path(raw_path).exists():
            continue
        try:
            img = Image.open(raw_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            text = " ".join(w for w in data.get("text", []) if w.strip())
            confs = [float(c) for c in data.get("conf", []) if float(c) > 0]
            total_len += len(text)
            total_conf += (sum(confs) / len(confs)) if confs else 50.0
            count += 1
        except Exception:
            continue

    if count == 0:
        return 200.0, 70.0

    return total_len / count, total_conf / count


def probe_video(video_path: Path) -> ProbeResult:
    """Analyze video characteristics. Takes ~30-60 seconds."""
    check_binaries(["ffmpeg", "ffprobe"])

    # Step 1: metadata
    duration = ffprobe_duration_seconds(video_path)
    if duration <= 0:
        raise RuntimeError(
            f"Video has no duration (0 seconds). Is the file corrupt? {video_path}"
        )
    info = ffprobe_video_info(video_path)
    log(
        f"{info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
        f"{format_duration(duration)}, audio {'detected' if info['has_audio'] else 'not found'}"
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="videopipe_probe_"))
    try:
        # Step 2: sample ~25 frames evenly
        sample_interval = max(1.0, duration / 25.0)
        sample_dir = tmpdir / "samples"
        sample_entries = extract_sample_frames(
            video_path,
            sample_dir,
            sample_interval=sample_interval,
        )
        log_verbose(
            f"Probe: sampled {len(sample_entries)} frames (interval={sample_interval:.1f}s)"
        )

        # Step 3: scene change detection
        scene_dir = tmpdir / "scenes"
        scene_entries = extract_scene_frames(
            video_path,
            scene_dir,
            scene_threshold=0.25,
        )
        duration_minutes = max(duration / 60.0, 0.01)
        scene_changes_per_minute = len(scene_entries) / duration_minutes

        # Step 4: text density on ~5 frames
        avg_text_len, avg_conf = _probe_text_density(sample_entries, max_frames=5)
        log(
            f"{len(sample_entries)} samples, {len(scene_entries)} scene changes "
            f"({scene_changes_per_minute:.1f}/min), text: {avg_text_len:.0f} chars avg"
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return ProbeResult(
        duration_seconds=duration,
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        scene_changes_per_minute=scene_changes_per_minute,
        avg_text_length=avg_text_len,
        avg_ocr_confidence=avg_conf,
        has_audio=info["has_audio"],
        sample_frame_count=len(sample_entries),
        scene_frame_count=len(scene_entries),
    )
