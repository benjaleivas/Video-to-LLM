from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .align import build_dataset, build_dataset_windows
from .events import parse_events_log
from .ffmpeg_utils import (
    extract_audio,
    extract_frames_at_timestamps,
    extract_sample_frames,
    extract_scene_frames,
)
from .frame_policy import (
    compute_gap_fill_timestamps,
    merge_entries_by_timestamp,
    required_anchor_timestamps,
)
from .quality import evaluate_quality_gates, summarize_quality_report
from .transcribe import segments_to_srt, transcribe_audio
from .utils import (
    check_binaries,
    coerce_max_seconds,
    ensure_dir,
    ffprobe_duration_seconds,
    log,
    relativize_paths,
    write_json,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _is_flag_explicit(flag_name: str, argv: list[str]) -> bool:
    return any(arg == flag_name or arg.startswith(flag_name + "=") for arg in argv)


def _apply_profile_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    if args.profile == "local_safe":
        if not _is_flag_explicit("--transcribe-provider", argv):
            args.transcribe_provider = "whisper"
        if not _is_flag_explicit("--ocr-provider", argv):
            args.ocr_provider = "tesseract"
        if not _is_flag_explicit("--ocr-fallback-provider", argv):
            args.ocr_fallback_provider = "none"
    elif args.profile == "quality_first":
        if not _is_flag_explicit("--transcribe-provider", argv):
            args.transcribe_provider = "assemblyai"
        if not _is_flag_explicit("--ocr-provider", argv):
            args.ocr_provider = "google"
        if not _is_flag_explicit("--ocr-fallback-provider", argv):
            args.ocr_fallback_provider = "azure"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videopipe",
        description="Local pipeline: video -> transcript + keyframes + OCR + aligned dataset",
    )
    parser.add_argument("--video", required=True, help="Input .mov/.mp4 path")
    parser.add_argument("--out", default="./output", help="Output root directory")
    parser.add_argument("--profile", choices=["local_safe", "quality_first"], default="quality_first")

    parser.add_argument("--scene-threshold", type=float, default=0.25, help="Scene threshold for ffmpeg select")
    parser.add_argument("--frame-policy", choices=["scene_only", "hybrid"], default="hybrid")
    parser.add_argument("--periodic-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-frame-gap-seconds", type=float, default=15.0)
    parser.add_argument("--always-include-start-end", type=str2bool, default=True)

    parser.add_argument("--transcribe-provider", choices=["whisper", "assemblyai"], default="whisper")
    parser.add_argument("--whisper-model", default="large-v3", help="faster-whisper model name")
    parser.add_argument("--force-language", default="en", help="Force language for transcription")
    parser.add_argument(
        "--assembly-diarization",
        type=str2bool,
        default=True,
        help="Enable AssemblyAI speaker diarization when --transcribe-provider assemblyai",
    )
    parser.add_argument(
        "--assembly-speaker-label-format",
        choices=["alpha", "numeric"],
        default="alpha",
        help="Speaker label format for AssemblyAI output",
    )
    parser.add_argument("--segment-max-seconds", type=float, default=25.0)
    parser.add_argument("--segment-silence-gap-seconds", type=float, default=0.6)
    parser.add_argument("--segment-min-seconds", type=float, default=3.0)

    parser.add_argument("--ocr-provider", choices=["tesseract", "google"], default="tesseract")
    parser.add_argument("--ocr-lang", default="eng", help="OCR language code")
    parser.add_argument("--ocr-scale", type=float, default=2.0, help="Frame upscale factor for OCR preprocessing")
    parser.add_argument(
        "--ocr-threshold",
        choices=["none", "otsu", "adaptive"],
        default="adaptive",
        help="Threshold strategy for OCR preprocessing",
    )
    parser.add_argument("--ocr-format", choices=["png", "jpg"], default="png", help="Processed frame format")
    parser.add_argument("--ocr-denoise", type=str2bool, default=True, help="Enable median denoise")
    parser.add_argument("--ocr-sharpen", type=str2bool, default=True, help="Enable unsharp mask")
    parser.add_argument("--ocr-crops", choices=["none", "preset"], default="preset", help="Crop mode for OCR")
    parser.add_argument(
        "--ocr-crop",
        action="append",
        default=[],
        help="Manual crop x1,y1,x2,y2 (repeatable)",
    )
    parser.add_argument("--psm", type=int, default=6, help="Tesseract psm for primary OCR pass")
    parser.add_argument(
        "--ocr-second-psm",
        type=int,
        default=None,
        help="Optional second psm; use when second pass likely improves sparse text",
    )
    parser.add_argument("--ocr-workers", type=int, default=4, help="Max OCR worker processes")
    parser.add_argument(
        "--google-ocr-feature",
        choices=["text_detection", "document_text_detection"],
        default="document_text_detection",
        help="Google Vision OCR feature when --ocr-provider google",
    )
    parser.add_argument("--google-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--google-max-concurrency", type=int, default=4)
    parser.add_argument("--ocr-quality-threshold", type=float, default=0.55)
    parser.add_argument("--ocr-fallback-provider", choices=["none", "azure"], default="azure")

    parser.add_argument("--dedupe-time-gap", type=float, default=2.0, help="Dedup compare window (seconds)")
    parser.add_argument("--dedupe-hash-threshold", type=int, default=6, help="Dedup phash hamming threshold")
    parser.add_argument("--dedupe-preserve-transitions", type=str2bool, default=True)
    parser.add_argument("--dedupe-transition-gap-seconds", type=float, default=2.5)

    parser.add_argument("--align-max-gap", type=float, default=10.0, help="Max seconds for segment->frame attach")
    parser.add_argument("--align-topk", type=int, default=3)
    parser.add_argument("--align-mode", choices=["nearest", "overlap_topk"], default="overlap_topk")

    parser.add_argument("--window-seconds", type=float, default=45.0, help="Window size for dataset_windows")
    parser.add_argument("--events-log", default="", help="Optional path to click/scroll/key event sidecar log")
    parser.add_argument("--enforce-quality", type=str2bool, default=False)

    parser.add_argument("--max-seconds", type=float, default=None, help="Trim processing duration for smoke tests")
    parser.add_argument("--max-minutes", type=float, default=None, help="Trim processing duration for smoke tests")
    parser.add_argument("--keep-raw-frames", type=str2bool, default=True, help="Keep frames_raw directory")
    return parser


def _clean_dir(path: Path) -> None:
    ensure_dir(path)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _check_provider_environment(args: argparse.Namespace) -> None:
    if args.transcribe_provider == "assemblyai":
        api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY is not set. Export it before using --transcribe-provider assemblyai"
            )

    if args.ocr_provider == "google":
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if not creds:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. Export it before using --ocr-provider google"
            )
        creds_path = Path(creds).expanduser().resolve()
        if not creds_path.exists() or not creds_path.is_file():
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS does not point to a readable file: "
                f"{creds_path}"
            )

    if args.ocr_provider == "google" and args.ocr_fallback_provider == "azure":
        endpoint = os.getenv("AZURE_VISION_ENDPOINT", "").strip()
        key = os.getenv("AZURE_VISION_KEY", "").strip()
        if not endpoint or not key:
            raise RuntimeError(
                "Azure fallback selected but AZURE_VISION_ENDPOINT or AZURE_VISION_KEY is missing"
            )


def _build_output_readme(video_path: Path, run_dir: Path, args: argparse.Namespace) -> str:
    return f"""# videopipe output

This folder contains processing artifacts for:

- video: `{video_path}`
- output root: `{run_dir}`

## Install prerequisites

```bash
brew install ffmpeg tesseract
python -m pip install -r requirements.txt
```

## Provider configuration used

- profile: `{args.profile}`
- frame policy: `{args.frame_policy}`
- transcribe provider: `{args.transcribe_provider}`
- OCR provider: `{args.ocr_provider}`
- OCR fallback provider: `{args.ocr_fallback_provider}`
- forced language: `{args.force_language}`
- Assembly diarization: `{args.assembly_diarization}`
- Google OCR feature: `{args.google_ocr_feature}`

## Example usage (quality-first)

```bash
python -m videopipe \
  --video ./input/video.mov \
  --profile quality_first \
  --frame-policy hybrid \
  --periodic-interval-seconds 15 \
  --max-frame-gap-seconds 15
```

## Generated files

- `audio.wav`
- `transcript.json`
- `transcript_utterances.json`
- `transcript.srt`
- `speakers.json`
- `events.json` (when `--events-log` provided)
- `frames_raw/`
- `frames/`
- `frames_index.json`
- `kept_frames_index.json`
- `dropped_frames.json`
- `frames_ocr.json`
- `dataset.json`
- `dataset_windows.json`
- `quality_report.json`
"""


def _pipeline(args: argparse.Namespace) -> int:
    start_time = time.time()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    _check_provider_environment(args)

    binaries = ["ffmpeg", "ffprobe"]
    if args.ocr_provider == "tesseract":
        binaries.append("tesseract")
    check_binaries(binaries)

    max_seconds = coerce_max_seconds(args.max_seconds, args.max_minutes)
    duration = ffprobe_duration_seconds(video_path)
    effective_duration = min(duration, max_seconds) if max_seconds is not None else duration

    out_root = Path(args.out).expanduser().resolve()
    run_dir = (out_root / video_path.stem).resolve()
    frames_raw_dir = run_dir / "frames_raw"
    frames_dir = run_dir / "frames"
    audio_path = run_dir / "audio.wav"

    ensure_dir(run_dir)
    _clean_dir(frames_raw_dir)
    _clean_dir(frames_dir)

    log(f"Video: {video_path}")
    log(f"Output directory: {run_dir}")
    log(f"Input duration: {duration:.2f}s | Processing duration: {effective_duration:.2f}s")

    log("Step 1/11: Extracting mono 16k audio")
    extract_audio(video_path, audio_path, max_seconds=max_seconds)

    log("Step 2/11: Extracting keyframe candidates")
    scene_entries = extract_scene_frames(
        video_path,
        frames_raw_dir=frames_raw_dir,
        scene_threshold=args.scene_threshold,
        max_seconds=max_seconds,
    )
    log(f"Scene frames found: {len(scene_entries)}")

    periodic_entries: list[dict] = []
    if args.frame_policy == "hybrid":
        periodic_entries = extract_sample_frames(
            video_path,
            frames_raw_dir=frames_raw_dir,
            sample_interval=args.periodic_interval_seconds,
            max_seconds=max_seconds,
        )
        log(f"Periodic frames extracted: {len(periodic_entries)}")

    candidate_entries = merge_entries_by_timestamp(scene_entries + periodic_entries, epsilon=0.2)
    anchor_ts = required_anchor_timestamps(
        candidate_entries,
        duration_seconds=effective_duration,
        always_include_start_end=bool(args.always_include_start_end),
        epsilon=0.2,
    )
    if anchor_ts:
        anchor_entries = extract_frames_at_timestamps(
            video_path,
            frames_raw_dir,
            anchor_ts,
            source="anchor",
            filename_prefix="anchor",
        )
        candidate_entries = merge_entries_by_timestamp(candidate_entries + anchor_entries, epsilon=0.2)
        log(f"Added anchor frames: {len(anchor_entries)}")

    if not candidate_entries:
        raise RuntimeError("No frames extracted from video. Try lowering --scene-threshold.")

    log("Step 3/11: Preprocessing frames for OCR")
    try:
        from .preprocess import PreprocessOptions, preprocess_frames
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Python preprocessing dependency. Run: pip install -r requirements.txt") from exc

    preprocess_options = PreprocessOptions(
        scale=args.ocr_scale,
        threshold=args.ocr_threshold,
        image_format=args.ocr_format,
        denoise=bool(args.ocr_denoise),
        sharpen=bool(args.ocr_sharpen),
    )
    processed_entries = preprocess_frames(candidate_entries, frames_out_dir=frames_dir, options=preprocess_options)

    log("Step 4/11: De-duplicating near-identical frames")
    try:
        from .ocr import OcrOptions, dedupe_frames, parse_manual_crops
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Python OCR dependency. Run: pip install -r requirements.txt") from exc

    kept_frames, dropped_frames = dedupe_frames(
        processed_entries,
        time_gap_sec=args.dedupe_time_gap,
        hamming_threshold=args.dedupe_hash_threshold,
        preserve_transitions=bool(args.dedupe_preserve_transitions),
        transition_gap_seconds=args.dedupe_transition_gap_seconds,
    )

    # Enforce max frame gap after dedupe by targeted backfill.
    gap_rounds = 0
    all_processed_entries = list(processed_entries)
    while gap_rounds < 3:
        gap_rounds += 1
        current_timestamps = [float(item["timestamp"]) for item in kept_frames]
        fill_ts = compute_gap_fill_timestamps(
            current_timestamps,
            duration_seconds=effective_duration,
            max_gap_seconds=args.max_frame_gap_seconds,
            epsilon=0.2,
            always_include_start_end=bool(args.always_include_start_end),
        )
        if not fill_ts:
            break

        log(f"Max-gap backfill round {gap_rounds}: extracting {len(fill_ts)} frame(s)")
        gap_entries = extract_frames_at_timestamps(
            video_path,
            frames_raw_dir,
            fill_ts,
            source="gap_fill",
            filename_prefix=f"gap{gap_rounds}",
        )
        if not gap_entries:
            break
        gap_processed = preprocess_frames(gap_entries, frames_out_dir=frames_dir, options=preprocess_options)
        all_processed_entries.extend(gap_processed)

        combined = merge_entries_by_timestamp(kept_frames + gap_processed, epsilon=0.05)
        kept_frames, dropped_round = dedupe_frames(
            combined,
            time_gap_sec=args.dedupe_time_gap,
            hamming_threshold=args.dedupe_hash_threshold,
            preserve_transitions=bool(args.dedupe_preserve_transitions),
            transition_gap_seconds=args.dedupe_transition_gap_seconds,
        )
        for item in dropped_round:
            item["dropped_phase"] = f"gap_backfill_round_{gap_rounds}"
        dropped_frames.extend(dropped_round)

    write_json(run_dir / "frames_index.json", relativize_paths(all_processed_entries, run_dir))
    write_json(run_dir / "kept_frames_index.json", relativize_paths(kept_frames, run_dir))
    write_json(run_dir / "dropped_frames.json", relativize_paths(dropped_frames, run_dir))
    log(
        f"Dedup result: kept={len(kept_frames)} dropped={len(dropped_frames)} "
        f"(time_gap={args.dedupe_time_gap}s, hash_threshold={args.dedupe_hash_threshold})"
    )

    log("Step 5/11: Transcribing audio")
    speakers: list[dict] = []
    transcript_utterances: list[dict] = []
    if args.transcribe_provider == "whisper":
        transcript_segments = transcribe_audio(
            audio_path,
            model_name=args.whisper_model,
            language=args.force_language,
            compute_type="int8",
        )
    else:
        from .transcribe_assemblyai import transcribe_audio_assemblyai

        transcript_segments, speakers, transcript_utterances = transcribe_audio_assemblyai(
            audio_path,
            language=args.force_language,
            diarization=bool(args.assembly_diarization),
            speaker_label_format=args.assembly_speaker_label_format,
            segment_max_seconds=args.segment_max_seconds,
            segment_silence_gap_seconds=args.segment_silence_gap_seconds,
            segment_min_seconds=args.segment_min_seconds,
        )

    write_json(run_dir / "transcript.json", transcript_segments)
    write_json(run_dir / "transcript_utterances.json", transcript_utterances)
    write_json(run_dir / "speakers.json", speakers)
    (run_dir / "transcript.srt").write_text(segments_to_srt(transcript_segments), encoding="utf-8")
    log(f"Transcript segments: {len(transcript_segments)}")

    log("Step 6/11: Running OCR on kept frames")
    manual_crops = parse_manual_crops(args.ocr_crop)
    ocr_options = OcrOptions(
        lang=args.ocr_lang,
        psm=args.psm,
        second_psm=args.ocr_second_psm,
        oem=1,
        crops_mode=args.ocr_crops,
        manual_crops=manual_crops,
    )
    if args.ocr_provider == "tesseract":
        from .ocr import ocr_frames

        ocr_results = ocr_frames(kept_frames, options=ocr_options, workers=args.ocr_workers)
    else:
        from .ocr_google import ocr_frames_google

        ocr_results = ocr_frames_google(
            kept_frames,
            options=ocr_options,
            feature=args.google_ocr_feature,
            timeout_seconds=args.google_timeout_seconds,
            max_concurrency=args.google_max_concurrency,
            quality_threshold=args.ocr_quality_threshold,
            fallback_provider=args.ocr_fallback_provider,
        )

    write_json(run_dir / "frames_ocr.json", relativize_paths(ocr_results, run_dir))

    events: list[dict] = []
    if args.events_log:
        events_path = Path(args.events_log).expanduser().resolve()
        if not events_path.exists():
            raise FileNotFoundError(f"events log does not exist: {events_path}")
        events = parse_events_log(events_path)
        write_json(run_dir / "events.json", events)

    log("Step 7/11: Aligning transcript segments with keyframes")
    dataset_records = build_dataset(
        transcript_segments,
        ocr_results,
        align_max_gap=args.align_max_gap,
        align_topk=args.align_topk,
        align_mode=args.align_mode,
    )
    write_json(run_dir / "dataset.json", relativize_paths(dataset_records, run_dir))

    log("Step 8/11: Building windowed dataset chunks")
    windows = build_dataset_windows(
        transcript_segments,
        ocr_results,
        window_seconds=args.window_seconds,
        events=events,
    )
    write_json(run_dir / "dataset_windows.json", relativize_paths(windows, run_dir))

    log("Step 9/11: Computing quality report")
    quality_report = summarize_quality_report(
        duration_seconds=effective_duration,
        kept_frames=kept_frames,
        transcript_segments=transcript_segments,
        dataset_records=dataset_records,
        frame_ocr_records=ocr_results,
        window_seconds=args.window_seconds,
    )
    quality_report["gate_results"] = evaluate_quality_gates(quality_report)
    write_json(run_dir / "quality_report.json", quality_report)

    if args.enforce_quality:
        failed = [gate for gate in quality_report["gate_results"] if not gate["passed"]]
        if failed:
            reasons = ", ".join(
                f"{item['name']} observed={item['observed']} target={item['operator']}{item['target']}"
                for item in failed
            )
            raise RuntimeError(f"Quality gates failed: {reasons}")

    log("Step 10/11: Finalizing outputs")
    (run_dir / "README.md").write_text(_build_output_readme(video_path, run_dir, args), encoding="utf-8")
    if not args.keep_raw_frames:
        shutil.rmtree(frames_raw_dir, ignore_errors=True)
        log("Removed raw frames directory (--keep-raw-frames false).")

    elapsed = time.time() - start_time
    log("Step 11/11: Completed")
    log(
        "Summary: "
        f"segments={len(transcript_segments)} | "
        f"speakers={len(speakers)} | "
        f"raw_frames={len(all_processed_entries)} | "
        f"kept_frames={len(kept_frames)} | "
        f"transcribe_provider={args.transcribe_provider} | "
        f"ocr_provider={args.ocr_provider} | "
        f"frame_policy={args.frame_policy} | "
        f"ocr_scale={args.ocr_scale} | "
        f"ocr_threshold={args.ocr_threshold} | "
        f"ocr_crops={args.ocr_crops} | "
        f"runtime={elapsed:.1f}s"
    )
    log(f"Output written to: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv_list = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv_list)
    _apply_profile_defaults(args, argv_list)
    try:
        return _pipeline(args)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1
