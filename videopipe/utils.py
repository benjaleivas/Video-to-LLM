from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(f"[videopipe] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    log(f"$ {shlex.join(cmd)}")
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=capture_output,
        cwd=str(cwd) if cwd else None,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}\n{detail}")
    return result


def check_binaries(names: list[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if not missing:
        return
    lines = ["Missing required binaries: " + ", ".join(missing)]
    if {"ffmpeg", "ffprobe", "tesseract"} & set(missing):
        lines.append("Install on macOS: brew install ffmpeg tesseract")
    raise FileNotFoundError("\n".join(lines))


def ffprobe_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = run_cmd(cmd, capture_output=True)
    value = (result.stdout or "").strip()
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse ffprobe duration: {value!r}") from exc


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_max_seconds(max_seconds: float | None, max_minutes: float | None) -> float | None:
    values: list[float] = []
    if max_seconds is not None:
        values.append(max_seconds)
    if max_minutes is not None:
        values.append(max_minutes * 60.0)
    if not values:
        return None
    return min(values)


def format_seconds_for_filename(seconds: float) -> str:
    return f"{seconds:010.3f}"


def to_relative(path_value: str | None, root: Path) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def relativize_paths(data: Any, root: Path) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if key.endswith("_path") and isinstance(value, str):
                out[key] = to_relative(value, root)
            else:
                out[key] = relativize_paths(value, root)
        return out
    if isinstance(data, list):
        return [relativize_paths(item, root) for item in data]
    return data


def format_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
