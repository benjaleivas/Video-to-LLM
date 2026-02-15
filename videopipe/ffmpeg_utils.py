from __future__ import annotations

import re
from pathlib import Path

from .utils import ensure_dir, log, run_cmd

PTS_TIME_RE = re.compile(r"pts_time:([0-9eE+\-.]+)")


def extract_audio(video_path: Path, audio_out: Path, max_seconds: float | None = None) -> None:
    ensure_dir(audio_out.parent)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(video_path)]
    if max_seconds is not None:
        cmd.extend(["-t", f"{max_seconds:.3f}"])
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000", str(audio_out)])
    run_cmd(cmd)


def _parse_pts_times(ffmpeg_stderr: str) -> list[float]:
    timestamps: list[float] = []
    for line in ffmpeg_stderr.splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        match = PTS_TIME_RE.search(line)
        if not match:
            continue
        try:
            timestamps.append(float(match.group(1)))
        except ValueError:
            continue
    return timestamps


def extract_scene_frames(
    video_path: Path,
    frames_raw_dir: Path,
    scene_threshold: float,
    max_seconds: float | None = None,
) -> list[dict]:
    ensure_dir(frames_raw_dir)
    out_pattern = frames_raw_dir / "frame_%06d.jpg"
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(video_path)]
    if max_seconds is not None:
        cmd.extend(["-t", f"{max_seconds:.3f}"])
    cmd.extend(
        [
            "-vf",
            f"select='gt(scene,{scene_threshold})',showinfo",
            "-vsync",
            "vfr",
            str(out_pattern),
        ]
    )
    result = run_cmd(cmd, capture_output=True)
    timestamps = _parse_pts_times(result.stderr or "")
    files = sorted(frames_raw_dir.glob("frame_*.jpg"))

    if not files:
        return []
    if len(files) != len(timestamps):
        log(
            "Warning: scene frame count and parsed timestamps differ "
            f"({len(files)} files vs {len(timestamps)} timestamps)."
        )

    entries: list[dict] = []
    for i, file_path in enumerate(files, start=1):
        timestamp = timestamps[i - 1] if i - 1 < len(timestamps) else float(i - 1)
        entries.append(
            {
                "timestamp": round(float(timestamp), 3),
                "raw_path": str(file_path.resolve()),
                "source": "scene",
                "original_index": i,
            }
        )
    return entries


def extract_sample_frames(
    video_path: Path,
    frames_raw_dir: Path,
    sample_interval: float = 3.0,
    max_seconds: float | None = None,
) -> list[dict]:
    ensure_dir(frames_raw_dir)
    out_pattern = frames_raw_dir / "sample_%06d.jpg"
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(video_path)]
    if max_seconds is not None:
        cmd.extend(["-t", f"{max_seconds:.3f}"])
    cmd.extend(
        [
            "-vf",
            f"fps=1/{sample_interval}",
            "-vsync",
            "vfr",
            str(out_pattern),
        ]
    )
    run_cmd(cmd)

    files = sorted(frames_raw_dir.glob("sample_*.jpg"))
    entries: list[dict] = []
    for i, file_path in enumerate(files, start=1):
        timestamp = (i - 1) * sample_interval
        entries.append(
            {
                "timestamp": round(float(timestamp), 3),
                "raw_path": str(file_path.resolve()),
                "source": "sample",
                "original_index": i,
            }
        )
    return entries


def merge_frame_entries(scene_entries: list[dict], sample_entries: list[dict], min_gap: float = 0.05) -> list[dict]:
    merged: list[dict] = []
    for entry in sorted(scene_entries + sample_entries, key=lambda item: item["timestamp"]):
        if merged and abs(entry["timestamp"] - merged[-1]["timestamp"]) < min_gap:
            continue
        merged.append({**entry, "original_index": len(merged) + 1})
    return merged
