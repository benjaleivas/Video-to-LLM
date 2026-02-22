from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console

_console = Console(highlight=False)
_log_file: TextIO | None = None
_error_file: TextIO | None = None


def init_logging(run_dir: Path) -> None:
    global _log_file, _error_file
    if _log_file is not None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_file = open(run_dir / "videopipe.log", "w", encoding="utf-8")
    _error_file = open(run_dir / "videopipe_errors.log", "w", encoding="utf-8")
    _write_file(f"=== videopipe log started {datetime.now().isoformat()} ===")


def close_logging() -> None:
    global _log_file, _error_file
    if _log_file:
        _log_file.close()
        _log_file = None
    if _error_file:
        _error_file.close()
        _error_file = None


def _write_file(message: str) -> None:
    if _log_file:
        _log_file.write(message + "\n")
        _log_file.flush()


def _write_error(message: str) -> None:
    _write_file(message)
    if _error_file:
        _error_file.write(message + "\n")
        _error_file.flush()


def log(message: str) -> None:
    _console.print(f"[dim]videopipe[/dim] {message}")
    _write_file(f"[videopipe] {message}")


def log_verbose(message: str) -> None:
    _write_file(f"[videopipe] {message}")


def log_error(message: str) -> None:
    _console.print(f"[bold red]ERROR[/bold red] {message}")
    _write_error(f"[ERROR] {message}")


def log_warning(message: str) -> None:
    _console.print(f"[yellow]WARN[/yellow]  {message}")
    _write_file(f"[WARN] {message}")


def log_section(title: str) -> None:
    _console.print()
    _console.print(f"[bold]=== {title} ===[/bold]")
    _write_file(f"\n=== {title} ===")


def get_console() -> Console:
    return _console


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    log_verbose(f"$ {shlex.join(cmd)}")
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
        log_error(f"Command failed ({result.returncode}): {shlex.join(cmd)}")
        log_verbose(detail)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(cmd)}\n{detail}"
        )
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


def coerce_max_seconds(
    max_seconds: float | None, max_minutes: float | None
) -> float | None:
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


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _to_relative(path_value: str, root: Path) -> str:
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
                out[key] = _to_relative(value, root)
            else:
                out[key] = relativize_paths(value, root)
        return out
    if isinstance(data, list):
        return [relativize_paths(item, root) for item in data]
    return data


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def make_progress(label: str) -> Any:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    return Progress(
        TextColumn("[dim]videopipe[/dim]"),
        TextColumn(label),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=True,
    )
