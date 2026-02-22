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
    close_logging,
    coerce_max_seconds,
    ensure_dir,
    ffprobe_duration_seconds,
    format_duration,
    get_console,
    init_logging,
    log,
    log_error,
    log_verbose,
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
    parser.add_argument(
        "--profile", choices=["local_safe", "quality_first"], default="quality_first"
    )

    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.25,
        help="Scene threshold for ffmpeg select",
    )
    parser.add_argument(
        "--frame-policy", choices=["scene_only", "hybrid"], default="hybrid"
    )
    parser.add_argument("--periodic-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-frame-gap-seconds", type=float, default=15.0)
    parser.add_argument("--always-include-start-end", type=str2bool, default=True)

    parser.add_argument(
        "--transcribe-provider", choices=["whisper", "assemblyai"], default="whisper"
    )
    parser.add_argument(
        "--whisper-model", default="large-v3", help="faster-whisper model name"
    )
    parser.add_argument(
        "--force-language", default="en", help="Force language for transcription"
    )
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

    parser.add_argument(
        "--ocr-provider", choices=["tesseract", "google"], default="tesseract"
    )
    parser.add_argument("--ocr-lang", default="eng", help="OCR language code")
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=2.0,
        help="Frame upscale factor for OCR preprocessing",
    )
    parser.add_argument(
        "--ocr-threshold",
        choices=["none", "otsu", "adaptive"],
        default="adaptive",
        help="Threshold strategy for OCR preprocessing",
    )
    parser.add_argument(
        "--ocr-format",
        choices=["png", "jpg"],
        default="png",
        help="Processed frame format",
    )
    parser.add_argument(
        "--ocr-denoise", type=str2bool, default=True, help="Enable median denoise"
    )
    parser.add_argument(
        "--ocr-sharpen", type=str2bool, default=True, help="Enable unsharp mask"
    )
    parser.add_argument(
        "--ocr-crops",
        choices=["none", "preset"],
        default="preset",
        help="Crop mode for OCR",
    )
    parser.add_argument(
        "--ocr-crop",
        action="append",
        default=[],
        help="Manual crop x1,y1,x2,y2 (repeatable)",
    )
    parser.add_argument(
        "--psm", type=int, default=6, help="Tesseract psm for primary OCR pass"
    )
    parser.add_argument(
        "--ocr-second-psm",
        type=int,
        default=None,
        help="Optional second psm; use when second pass likely improves sparse text",
    )
    parser.add_argument(
        "--ocr-workers", type=int, default=4, help="Max OCR worker processes"
    )
    parser.add_argument(
        "--google-ocr-feature",
        choices=["text_detection", "document_text_detection"],
        default="document_text_detection",
        help="Google Vision OCR feature when --ocr-provider google",
    )
    parser.add_argument("--google-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--google-max-concurrency", type=int, default=4)
    parser.add_argument("--ocr-quality-threshold", type=float, default=0.55)
    parser.add_argument(
        "--ocr-fallback-provider", choices=["none", "azure"], default="azure"
    )

    parser.add_argument(
        "--dedupe-time-gap",
        type=float,
        default=2.0,
        help="Dedup compare window (seconds)",
    )
    parser.add_argument(
        "--dedupe-hash-threshold",
        type=int,
        default=6,
        help="Dedup phash hamming threshold",
    )
    parser.add_argument("--dedupe-preserve-transitions", type=str2bool, default=True)
    parser.add_argument("--dedupe-transition-gap-seconds", type=float, default=2.5)

    parser.add_argument(
        "--align-max-gap",
        type=float,
        default=10.0,
        help="Max seconds for segment->frame attach",
    )
    parser.add_argument("--align-topk", type=int, default=3)
    parser.add_argument(
        "--align-mode", choices=["nearest", "overlap_topk"], default="overlap_topk"
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=45.0,
        help="Window size for dataset_windows",
    )
    parser.add_argument(
        "--events-log",
        default="",
        help="Optional path to click/scroll/key event sidecar log",
    )
    parser.add_argument("--enforce-quality", type=str2bool, default=False)

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Trim processing duration for smoke tests",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Trim processing duration for smoke tests",
    )
    parser.add_argument(
        "--keep-raw-frames",
        type=str2bool,
        default=True,
        help="Keep frames_raw directory",
    )
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
        if args.ocr_fallback_provider == "azure":
            endpoint = os.getenv("AZURE_VISION_ENDPOINT", "").strip()
            key = os.getenv("AZURE_VISION_KEY", "").strip()
            if not endpoint or not key:
                raise RuntimeError(
                    "Azure fallback selected but AZURE_VISION_ENDPOINT or AZURE_VISION_KEY is missing"
                )


def _build_output_readme(
    video_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    num_segments: int = 0,
    num_speakers: int = 0,
    num_kept_frames: int = 0,
    elapsed: float = 0.0,
) -> str:
    return f"""# videopipe output

Processed **{video_path.name}** in {format_duration(elapsed)}.

## How to use with an LLM

The primary output is `dataset_windows.json` — time-windowed chunks containing
aligned transcript text, OCR text from on-screen content, and references to
keyframe images. Feed these windows to any LLM to ask questions about the video.

## Summary

| Metric | Value |
|--------|-------|
| Transcript segments | {num_segments} |
| Speakers detected | {num_speakers} |
| Keyframes kept | {num_kept_frames} |
| Transcription | {args.transcribe_provider} |
| OCR | {args.ocr_provider} |
| OCR fallback | {args.ocr_fallback_provider} |
| Profile | {args.profile} |

## Generated files

| File | Description |
|------|-------------|
| `dataset_windows.json` | **LLM-ready** windowed chunks (transcript + OCR + frame refs) |
| `dataset.json` | Per-segment aligned records |
| `transcript.json` | Timed transcript segments |
| `transcript.srt` | SubRip subtitles |
| `transcript_utterances.json` | Speaker-attributed utterances |
| `speakers.json` | Speaker metadata |
| `frames/` | Preprocessed keyframes (OCR-optimized) |
| `frames_raw/` | Original extracted frames |
| `frames_ocr.json` | OCR results per frame with quality scores |
| `kept_frames_index.json` | Index of deduplicated frames |
| `dropped_frames.json` | Frames removed by deduplication |
| `quality_report.json` | Coverage, alignment, and OCR quality metrics |
| `videopipe.log` | Full processing log |
| `videopipe_errors.log` | Errors only (empty on success) |

## Reproduce this run

```bash
python -m videopipe \\
  --video {video_path} \\
  --out {run_dir.parent} \\
  --profile {args.profile} \\
  --transcribe-provider {args.transcribe_provider} \\
  --ocr-provider {args.ocr_provider} \\
  --ocr-fallback-provider {args.ocr_fallback_provider} \\
  --frame-policy {args.frame_policy} \\
  --periodic-interval-seconds {args.periodic_interval_seconds} \\
  --max-frame-gap-seconds {args.max_frame_gap_seconds}
```
"""


def _step_log(step: int, total: int, description: str) -> None:
    log(f"Step {step:>2}/{total}  {description}")


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
    effective_duration = (
        min(duration, max_seconds) if max_seconds is not None else duration
    )

    out_root = Path(args.out).expanduser().resolve()
    run_dir = (out_root / video_path.stem).resolve()
    frames_raw_dir = run_dir / "frames_raw"
    frames_dir = run_dir / "frames"
    audio_path = run_dir / "audio.wav"

    ensure_dir(run_dir)
    init_logging(run_dir)
    _clean_dir(frames_raw_dir)
    _clean_dir(frames_dir)

    log(f"Video: {video_path}")
    log(f"Output: {run_dir}")
    log(f"Duration: {format_duration(duration)}")
    log_verbose(f"Effective duration: {effective_duration:.2f}s")

    TOTAL_STEPS = 11

    # -- Step 1: Audio extraction --
    _step_log(1, TOTAL_STEPS, "Extracting audio")
    extract_audio(video_path, audio_path, max_seconds=max_seconds)

    # -- Step 2: Keyframe extraction --
    _step_log(2, TOTAL_STEPS, "Extracting keyframes")
    scene_entries = extract_scene_frames(
        video_path,
        frames_raw_dir=frames_raw_dir,
        scene_threshold=args.scene_threshold,
        max_seconds=max_seconds,
    )

    periodic_entries: list[dict] = []
    if args.frame_policy == "hybrid":
        periodic_entries = extract_sample_frames(
            video_path,
            frames_raw_dir=frames_raw_dir,
            sample_interval=args.periodic_interval_seconds,
            max_seconds=max_seconds,
        )

    candidate_entries = merge_entries_by_timestamp(
        scene_entries + periodic_entries, epsilon=0.2
    )
    anchor_ts = required_anchor_timestamps(
        candidate_entries,
        duration_seconds=effective_duration,
        always_include_start_end=args.always_include_start_end,
        epsilon=0.2,
    )
    anchor_count = 0
    if anchor_ts:
        anchor_entries = extract_frames_at_timestamps(
            video_path,
            frames_raw_dir,
            anchor_ts,
            source="anchor",
            filename_prefix="anchor",
        )
        candidate_entries = merge_entries_by_timestamp(
            candidate_entries + anchor_entries, epsilon=0.2
        )
        anchor_count = len(anchor_entries)

    if not candidate_entries:
        raise RuntimeError(
            "No frames extracted from video. Try lowering --scene-threshold."
        )

    log(
        f"  \u2192 {len(scene_entries)} scene + {len(periodic_entries)} periodic"
        + (f" + {anchor_count} anchor" if anchor_count else "")
        + " frames"
    )

    # -- Step 3: Preprocessing --
    _step_log(3, TOTAL_STEPS, "Preprocessing frames")
    try:
        from .preprocess import PreprocessOptions, preprocess_frames
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python preprocessing dependency. Run: pip install -r requirements.txt"
        ) from exc

    preprocess_options = PreprocessOptions(
        scale=args.ocr_scale,
        threshold=args.ocr_threshold,
        image_format=args.ocr_format,
        denoise=args.ocr_denoise,
        sharpen=args.ocr_sharpen,
    )
    processed_entries = preprocess_frames(
        candidate_entries, frames_out_dir=frames_dir, options=preprocess_options
    )

    # -- Step 4: Deduplication --
    _step_log(4, TOTAL_STEPS, "Deduplicating frames")
    try:
        from .ocr import OcrOptions, dedupe_frames, parse_manual_crops
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python OCR dependency. Run: pip install -r requirements.txt"
        ) from exc

    kept_frames, dropped_frames = dedupe_frames(
        processed_entries,
        time_gap_sec=args.dedupe_time_gap,
        hamming_threshold=args.dedupe_hash_threshold,
        preserve_transitions=args.dedupe_preserve_transitions,
        transition_gap_seconds=args.dedupe_transition_gap_seconds,
    )

    all_processed_entries = list(processed_entries)
    for gap_round in range(1, 4):
        current_timestamps = [float(item["timestamp"]) for item in kept_frames]
        fill_ts = compute_gap_fill_timestamps(
            current_timestamps,
            duration_seconds=effective_duration,
            max_gap_seconds=args.max_frame_gap_seconds,
            epsilon=0.2,
            always_include_start_end=args.always_include_start_end,
        )
        if not fill_ts:
            break

        log_verbose(
            f"Max-gap backfill round {gap_round}: extracting {len(fill_ts)} frame(s)"
        )
        gap_entries = extract_frames_at_timestamps(
            video_path,
            frames_raw_dir,
            fill_ts,
            source="gap_fill",
            filename_prefix=f"gap{gap_round}",
        )
        if not gap_entries:
            break
        gap_processed = preprocess_frames(
            gap_entries, frames_out_dir=frames_dir, options=preprocess_options
        )
        all_processed_entries.extend(gap_processed)

        combined = merge_entries_by_timestamp(kept_frames + gap_processed, epsilon=0.05)
        kept_frames, dropped_round = dedupe_frames(
            combined,
            time_gap_sec=args.dedupe_time_gap,
            hamming_threshold=args.dedupe_hash_threshold,
            preserve_transitions=args.dedupe_preserve_transitions,
            transition_gap_seconds=args.dedupe_transition_gap_seconds,
        )
        for item in dropped_round:
            item["dropped_phase"] = f"gap_backfill_round_{gap_round}"
        dropped_frames.extend(dropped_round)

    write_json(
        run_dir / "frames_index.json", relativize_paths(all_processed_entries, run_dir)
    )
    write_json(
        run_dir / "kept_frames_index.json", relativize_paths(kept_frames, run_dir)
    )
    write_json(
        run_dir / "dropped_frames.json", relativize_paths(dropped_frames, run_dir)
    )
    log(f"  \u2192 kept {len(kept_frames)}, dropped {len(dropped_frames)}")

    # -- Step 5: Transcription --
    _step_log(5, TOTAL_STEPS, f"Transcribing audio ({args.transcribe_provider})")
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

        transcript_segments, speakers, transcript_utterances = (
            transcribe_audio_assemblyai(
                audio_path,
                language=args.force_language,
                diarization=bool(args.assembly_diarization),
                speaker_label_format=args.assembly_speaker_label_format,
                segment_max_seconds=args.segment_max_seconds,
                segment_silence_gap_seconds=args.segment_silence_gap_seconds,
                segment_min_seconds=args.segment_min_seconds,
            )
        )

    write_json(run_dir / "transcript.json", transcript_segments)
    write_json(run_dir / "transcript_utterances.json", transcript_utterances)
    write_json(run_dir / "speakers.json", speakers)
    (run_dir / "transcript.srt").write_text(
        segments_to_srt(transcript_segments), encoding="utf-8"
    )
    speaker_info = f", {len(speakers)} speakers" if speakers else ""
    log(f"  \u2192 {len(transcript_segments)} segments{speaker_info}")

    # -- Step 6: OCR --
    _step_log(6, TOTAL_STEPS, f"Running OCR ({args.ocr_provider})")
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

        ocr_results = ocr_frames(
            kept_frames, options=ocr_options, workers=args.ocr_workers
        )
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

    # -- Step 7: Alignment --
    _step_log(7, TOTAL_STEPS, "Aligning transcript + frames")
    dataset_records = build_dataset(
        transcript_segments,
        ocr_results,
        align_max_gap=args.align_max_gap,
        align_topk=args.align_topk,
        align_mode=args.align_mode,
    )
    write_json(run_dir / "dataset.json", relativize_paths(dataset_records, run_dir))

    # -- Step 8: Windowed dataset --
    _step_log(8, TOTAL_STEPS, "Building windowed dataset")
    windows = build_dataset_windows(
        transcript_segments,
        ocr_results,
        window_seconds=args.window_seconds,
        events=events,
    )
    write_json(run_dir / "dataset_windows.json", relativize_paths(windows, run_dir))

    # -- Step 9: Quality report --
    _step_log(9, TOTAL_STEPS, "Computing quality report")
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

    # -- Step 10: Finalize --
    _step_log(10, TOTAL_STEPS, "Finalizing outputs")
    elapsed = time.time() - start_time
    (run_dir / "README.md").write_text(
        _build_output_readme(
            video_path,
            run_dir,
            args,
            num_segments=len(transcript_segments),
            num_speakers=len(speakers),
            num_kept_frames=len(kept_frames),
            elapsed=elapsed,
        ),
        encoding="utf-8",
    )
    if not args.keep_raw_frames:
        shutil.rmtree(frames_raw_dir, ignore_errors=True)
        log_verbose("Removed raw frames directory (--keep-raw-frames false)")

    # -- Step 11: Done --
    log("")
    get_console().print(
        f"[bold green]\u2713[/bold green] Complete in {format_duration(elapsed)}"
    )
    log(
        f"  segments={len(transcript_segments)}  "
        f"frames={len(kept_frames)}  "
        f"speakers={len(speakers)}"
    )
    log(f"  Output: {run_dir}")
    log(f"  Log:    {run_dir / 'videopipe.log'}")

    log_verbose(
        f"Summary: segments={len(transcript_segments)} speakers={len(speakers)} "
        f"raw_frames={len(all_processed_entries)} kept_frames={len(kept_frames)} "
        f"transcribe_provider={args.transcribe_provider} ocr_provider={args.ocr_provider} "
        f"frame_policy={args.frame_policy} ocr_scale={args.ocr_scale} "
        f"ocr_threshold={args.ocr_threshold} ocr_crops={args.ocr_crops} "
        f"runtime={elapsed:.1f}s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv_list = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv_list)
    _apply_profile_defaults(args, argv_list)

    out_root = Path(args.out).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    run_dir = (out_root / video_path.stem).resolve()
    init_logging(run_dir)

    try:
        return _pipeline(args)
    except Exception as exc:
        log_error(str(exc))
        return 1
    finally:
        close_logging()
