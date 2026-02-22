"""Parameter tuner: convert probe results into optimal pipeline parameters.

Takes a ProbeResult from probe.py and returns an argparse.Namespace
compatible with cli._pipeline().
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .probe import ProbeResult
from .utils import log


def _ping_google_vision() -> bool:
    """Send a tiny test image to Google Vision to verify credentials and API access.

    Returns True if the API is reachable with valid auth. We send a 1x1 PNG
    which may trigger INVALID_ARGUMENT (code 3) — that's fine, it means auth
    and billing worked. Only fail on auth/permission/service errors.
    """
    # gRPC codes that mean "API is reachable, just didn't like our input"
    _OK_CODES = {0, 3}  # OK, INVALID_ARGUMENT

    try:
        import warnings

        from google.cloud import vision

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*end user credentials.*")
            client = vision.ImageAnnotatorClient()
            # 1x1 white PNG — minimal payload to validate auth + API access
            image = vision.Image(
                content=(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
                    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
                )
            )
            resp = client.text_detection(image=image, timeout=10)
        if resp.error.code in _OK_CODES:
            return True
        log(f"Tuner: Google Vision ping failed: {resp.error.message[:120]}")
        return False
    except Exception as exc:
        log(f"Tuner: Google Vision ping failed: {str(exc)[:120]}")
        return False


def _detect_providers() -> dict[str, str]:
    """Check which cloud API keys are available and pick the best stack.

    For Google Vision, goes beyond file-exists check: sends a real API ping
    to catch expired tokens, missing quota projects, and disabled APIs early.
    """
    # Load .env if present (won't override existing env vars)
    try:
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
            log(f"Tuner: loaded .env from {dotenv_path}")
    except ImportError:
        pass

    has_assemblyai = bool(os.getenv("ASSEMBLYAI_API_KEY", "").strip())
    has_google = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip())
    has_azure = bool(os.getenv("AZURE_VISION_ENDPOINT", "").strip()) and bool(
        os.getenv("AZURE_VISION_KEY", "").strip()
    )

    # Google creds: verify the file exists
    if has_google:
        creds_path = (
            Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")).expanduser().resolve()
        )
        if not creds_path.exists() or not creds_path.is_file():
            has_google = False

    # Google creds: validate actual API access (catches expired tokens,
    # missing quota project, disabled APIs, permission errors)
    if has_google:
        log("Tuner: validating Google Vision API access...")
        if _ping_google_vision():
            log("Tuner: Google Vision API OK")
        else:
            log("Tuner: Google Vision unavailable, falling back to tesseract")
            has_google = False

    transcribe = "assemblyai" if has_assemblyai else "whisper"
    ocr = "google" if has_google else "tesseract"
    ocr_fallback = "azure" if (has_google and has_azure) else "none"

    return {
        "transcribe_provider": transcribe,
        "ocr_provider": ocr,
        "ocr_fallback_provider": ocr_fallback,
    }


def _tune_frame_capture(probe: ProbeResult) -> dict:
    """Determine frame capture rate and policy from scene change frequency and duration."""
    scpm = probe.scene_changes_per_minute

    # Base interval from scene change frequency
    if scpm > 6:
        interval = 5.0
    elif scpm > 2:
        interval = 10.0
    else:
        interval = 20.0

    # Short videos: capture more aggressively
    if probe.duration_seconds < 120:
        interval = max(3.0, min(interval, 5.0))

    # Long videos: relax slightly
    if probe.duration_seconds > 1800:
        interval = max(interval, 15.0)

    # Frame policy: always hybrid for better coverage
    frame_policy = "hybrid"

    # Max gap tracks the interval
    max_gap = interval

    return {
        "periodic_interval_seconds": interval,
        "frame_policy": frame_policy,
        "max_frame_gap_seconds": max_gap,
    }


def _tune_ocr(probe: ProbeResult) -> dict:
    """Determine OCR preprocessing from resolution and text density."""
    # Scale factor based on resolution
    height = probe.height
    if height < 720:
        ocr_scale = 3.0
    elif height <= 1080:
        ocr_scale = 2.0
    else:
        ocr_scale = 1.5

    # Crop mode based on text density
    ocr_crops = "none" if probe.avg_text_length < 100 else "preset"

    return {
        "ocr_scale": ocr_scale,
        "ocr_crops": ocr_crops,
    }


def _tune_dedup(probe: ProbeResult) -> dict:
    """Determine dedup sensitivity from duration and scene activity."""
    # Longer videos need looser dedup to avoid keeping too many near-identical frames
    if probe.duration_seconds > 1800:
        time_gap = 3.0
        hash_threshold = 8
    elif probe.duration_seconds < 120:
        time_gap = 1.5
        hash_threshold = 5
    else:
        time_gap = 2.0
        hash_threshold = 6

    return {
        "dedupe_time_gap": time_gap,
        "dedupe_hash_threshold": hash_threshold,
    }


def tune_parameters(
    probe: ProbeResult, video_path: str, output_dir: str
) -> argparse.Namespace:
    """Convert probe results into a full argparse.Namespace for _pipeline().

    Bootstraps all defaults from build_parser() so any new CLI arguments are
    automatically inherited. Then overlays auto-tuned values.
    """
    from .cli import build_parser

    providers = _detect_providers()
    frame_params = _tune_frame_capture(probe)
    ocr_params = _tune_ocr(probe)
    dedup_params = _tune_dedup(probe)

    log(
        f"Tuner: providers -> transcribe={providers['transcribe_provider']}, "
        f"ocr={providers['ocr_provider']}, fallback={providers['ocr_fallback_provider']}"
    )
    log(
        f"Tuner: frames -> policy={frame_params['frame_policy']}, "
        f"interval={frame_params['periodic_interval_seconds']}s, "
        f"max_gap={frame_params['max_frame_gap_seconds']}s"
    )
    log(
        f"Tuner: ocr -> scale={ocr_params['ocr_scale']}, crops={ocr_params['ocr_crops']}"
    )
    log(
        f"Tuner: dedup -> time_gap={dedup_params['dedupe_time_gap']}s, "
        f"hash_threshold={dedup_params['dedupe_hash_threshold']}"
    )

    # Bootstrap from build_parser() defaults — any new CLI arg is auto-inherited
    args = build_parser().parse_args(["--video", video_path, "--out", output_dir])

    # Overlay auto-tuned values
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
