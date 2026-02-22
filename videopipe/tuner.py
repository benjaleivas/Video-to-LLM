"""Parameter tuner: convert probe results into optimal pipeline parameters.

Takes a ProbeResult from probe.py and returns an argparse.Namespace
compatible with cli._pipeline().
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.table import Table

from .probe import ProbeResult
from .utils import get_console, log, log_verbose, log_warning

_1X1_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _ping_google_vision() -> bool:
    """Send a tiny test image to Google Vision to verify credentials and API access.

    Returns True if the API is reachable with valid auth. We send a 1x1 PNG
    which may trigger INVALID_ARGUMENT (code 3) — that's fine, it means auth
    and billing worked. Only fail on auth/permission/service errors.
    """
    _OK_CODES = {0, 3}  # OK, INVALID_ARGUMENT

    try:
        import warnings

        from google.cloud import vision

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*end user credentials.*")
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=_1X1_PNG)
            resp = client.text_detection(image=image, timeout=10)
        if resp.error.code in _OK_CODES:
            return True
        log_verbose(f"Google Vision ping failed: {resp.error.message[:120]}")
        return False
    except Exception as exc:
        log_verbose(f"Google Vision ping failed: {str(exc)[:120]}")
        return False


def _ping_assemblyai() -> bool:
    """Verify AssemblyAI API key with a lightweight list request."""
    try:
        import requests

        api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        resp = requests.get(
            "https://api.assemblyai.com/v2/transcript",
            headers={"Authorization": api_key},
            params={"limit": 1},
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        log_verbose(f"AssemblyAI ping returned HTTP {resp.status_code}")
        return False
    except Exception as exc:
        log_verbose(f"AssemblyAI ping failed: {str(exc)[:120]}")
        return False


def _ping_azure_vision() -> bool:
    """Verify Azure Vision credentials with a minimal analyze request."""
    try:
        import requests

        endpoint = os.getenv("AZURE_VISION_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_VISION_KEY", "").strip()
        url = (
            endpoint.rstrip("/")
            + "/computervision/imageanalysis:analyze"
            + "?api-version=2023-10-01&features=read"
        )
        resp = requests.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "Content-Type": "application/octet-stream",
            },
            data=_1X1_PNG,
            timeout=10,
        )
        # 200 = OK, 400 = bad image but auth worked
        if resp.status_code in (200, 400):
            return True
        log_verbose(f"Azure Vision ping returned HTTP {resp.status_code}")
        return False
    except Exception as exc:
        log_verbose(f"Azure Vision ping failed: {str(exc)[:120]}")
        return False


def _detect_providers() -> dict[str, str]:
    """Check which cloud API keys are available, validate with pings, pick the best stack."""
    # Load .env if present (won't override existing env vars)
    try:
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            pkg_env = Path(__file__).resolve().parent.parent / ".env"
            if pkg_env.is_file():
                dotenv_path = str(pkg_env)
        if dotenv_path:
            load_dotenv(dotenv_path)
            log(f"Loaded .env from {dotenv_path}")
    except ImportError:
        pass

    has_assemblyai = bool(os.getenv("ASSEMBLYAI_API_KEY", "").strip())
    has_google = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip())
    has_azure = bool(os.getenv("AZURE_VISION_ENDPOINT", "").strip()) and bool(
        os.getenv("AZURE_VISION_KEY", "").strip()
    )

    # Validate AssemblyAI
    if has_assemblyai:
        log("Validating AssemblyAI API key...")
        if _ping_assemblyai():
            log("AssemblyAI API [green]OK[/green]")
        else:
            log_warning("AssemblyAI key invalid, falling back to Whisper")
            has_assemblyai = False

    # Google creds: verify file exists then ping API
    if has_google:
        creds_path = (
            Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")).expanduser().resolve()
        )
        if not creds_path.exists() or not creds_path.is_file():
            has_google = False

    if has_google:
        log("Validating Google Vision API access...")
        if _ping_google_vision():
            log("Google Vision API [green]OK[/green]")
        else:
            log_warning("Google Vision unavailable, falling back to Tesseract")
            has_google = False

    # Validate Azure Vision
    if has_azure:
        log("Validating Azure Vision API access...")
        if _ping_azure_vision():
            log("Azure Vision API [green]OK[/green]")
        else:
            log_warning("Azure Vision unavailable, disabling fallback")
            has_azure = False

    transcribe = "assemblyai" if has_assemblyai else "whisper"
    ocr = "google" if has_google else "tesseract"
    ocr_fallback = "azure" if (has_google and has_azure) else "none"

    return {
        "transcribe_provider": transcribe,
        "ocr_provider": ocr,
        "ocr_fallback_provider": ocr_fallback,
    }


def _tune_frame_capture(probe: ProbeResult) -> dict:
    scpm = probe.scene_changes_per_minute

    if scpm > 6:
        interval = 5.0
    elif scpm > 2:
        interval = 10.0
    else:
        interval = 20.0

    if probe.duration_seconds < 120:
        interval = max(3.0, min(interval, 5.0))

    if probe.duration_seconds > 1800:
        interval = max(interval, 15.0)

    return {
        "periodic_interval_seconds": interval,
        "frame_policy": "hybrid",
        "max_frame_gap_seconds": interval,
    }


def _tune_ocr(probe: ProbeResult) -> dict:
    height = probe.height
    if height < 720:
        ocr_scale = 3.0
    elif height <= 1080:
        ocr_scale = 2.0
    else:
        ocr_scale = 1.5

    ocr_crops = "none" if probe.avg_text_length < 100 else "preset"

    return {"ocr_scale": ocr_scale, "ocr_crops": ocr_crops}


def _tune_dedup(probe: ProbeResult) -> dict:
    if probe.duration_seconds > 1800:
        time_gap = 3.0
        hash_threshold = 8
    elif probe.duration_seconds < 120:
        time_gap = 1.5
        hash_threshold = 5
    else:
        time_gap = 2.0
        hash_threshold = 6

    return {"dedupe_time_gap": time_gap, "dedupe_hash_threshold": hash_threshold}


# -- reason helpers for the comparison table --


def _frame_reason(probe: ProbeResult) -> str:
    scpm = probe.scene_changes_per_minute
    if scpm > 6:
        return f"high scene rate ({scpm:.1f}/min)"
    if scpm > 2:
        return f"moderate scene rate ({scpm:.1f}/min)"
    return f"low scene rate ({scpm:.1f}/min)"


def _ocr_scale_reason(probe: ProbeResult) -> str:
    if probe.height < 720:
        return f"low res ({probe.height}p)"
    if probe.height <= 1080:
        return f"HD ({probe.height}p)"
    return f"high res ({probe.height}p)"


def _ocr_crops_reason(probe: ProbeResult) -> str:
    if probe.avg_text_length < 100:
        return "sparse text detected"
    return "dense text detected"


def _dedup_reason(probe: ProbeResult) -> str:
    if probe.duration_seconds > 1800:
        return "long video (>30min)"
    if probe.duration_seconds < 120:
        return "short video (<2min)"
    return "standard duration"


def tune_parameters(
    probe: ProbeResult, video_path: str, output_dir: str
) -> argparse.Namespace:
    """Convert probe results into a full argparse.Namespace for _pipeline()."""
    from .cli import build_parser

    providers = _detect_providers()
    frame_params = _tune_frame_capture(probe)
    ocr_params = _tune_ocr(probe)
    dedup_params = _tune_dedup(probe)

    # Get parser defaults for comparison
    defaults = build_parser().parse_args(["--video", video_path, "--out", output_dir])

    # Build the comparison table
    table = Table(
        title="Auto-Tuned Parameters",
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Parameter", style="cyan", min_width=16)
    table.add_column("Default", style="dim")
    table.add_column("Tuned", style="green")
    table.add_column("Reason", style="dim italic")

    rows = [
        (
            "Transcription",
            defaults.transcribe_provider,
            providers["transcribe_provider"],
            (
                "API key validated"
                if providers["transcribe_provider"] == "assemblyai"
                else "no API key"
            ),
        ),
        (
            "OCR",
            defaults.ocr_provider,
            providers["ocr_provider"],
            (
                "Google Vision validated"
                if providers["ocr_provider"] == "google"
                else "no API key"
            ),
        ),
        (
            "OCR fallback",
            defaults.ocr_fallback_provider,
            providers["ocr_fallback_provider"],
            (
                "Azure validated"
                if providers["ocr_fallback_provider"] == "azure"
                else "disabled"
            ),
        ),
        (
            "Frame interval",
            f"{defaults.periodic_interval_seconds}s",
            f"{frame_params['periodic_interval_seconds']}s",
            _frame_reason(probe),
        ),
        (
            "Max frame gap",
            f"{defaults.max_frame_gap_seconds}s",
            f"{frame_params['max_frame_gap_seconds']}s",
            "tracks interval",
        ),
        (
            "OCR scale",
            f"{defaults.ocr_scale}x",
            f"{ocr_params['ocr_scale']}x",
            _ocr_scale_reason(probe),
        ),
        (
            "OCR crops",
            defaults.ocr_crops,
            ocr_params["ocr_crops"],
            _ocr_crops_reason(probe),
        ),
        (
            "Dedup time gap",
            f"{defaults.dedupe_time_gap}s",
            f"{dedup_params['dedupe_time_gap']}s",
            _dedup_reason(probe),
        ),
        (
            "Dedup threshold",
            str(defaults.dedupe_hash_threshold),
            str(dedup_params["dedupe_hash_threshold"]),
            "",
        ),
    ]

    for label, default_val, tuned_val, reason in rows:
        changed = str(default_val) != str(tuned_val)
        style = "bold green" if changed else "dim"
        table.add_row(label, str(default_val), f"[{style}]{tuned_val}[/]", reason)

    get_console().print()
    get_console().print(table)

    # Log raw values to file
    log_verbose(f"Tuner: providers -> {providers}")
    log_verbose(f"Tuner: frames -> {frame_params}")
    log_verbose(f"Tuner: ocr -> {ocr_params}")
    log_verbose(f"Tuner: dedup -> {dedup_params}")

    # Overlay tuned values onto defaults (same Namespace object — safe because
    # the comparison table above has already been rendered from `defaults`).
    args = defaults

    args.profile = "auto"
    args.transcribe_provider = providers["transcribe_provider"]
    args.ocr_provider = providers["ocr_provider"]
    args.ocr_fallback_provider = providers["ocr_fallback_provider"]
    args.frame_policy = frame_params["frame_policy"]
    args.periodic_interval_seconds = frame_params["periodic_interval_seconds"]
    args.max_frame_gap_seconds = frame_params["max_frame_gap_seconds"]
    args.ocr_scale = ocr_params["ocr_scale"]
    args.ocr_crops = ocr_params["ocr_crops"]
    args.dedupe_time_gap = dedup_params["dedupe_time_gap"]
    args.dedupe_hash_threshold = dedup_params["dedupe_hash_threshold"]

    return args
