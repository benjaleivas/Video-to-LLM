# video-to-llm

`video-to-llm` is a pipeline for turning **web-app screen recordings** into structured, LLM-ready data.

It is optimized for analyst/product walkthrough videos where UI text is dense and small (dashboards, filters, tables, side panels), and where you need to map what was said to what was visible on screen.

## What It Produces

Given one video input, it outputs:
- clean audio (`audio.wav`)
- transcript with timing (+ optional diarization)
- keyframes with OCR text (including region crops)
- transcript-to-frame alignment records
- windowed dataset chunks for retrieval/analysis
- quality report with explicit gate checks

## Why This Is Different

Compared to naive “sample every N seconds” pipelines, this uses:
- hybrid keyframe policy: scene changes + periodic sampling + start/end anchors + max-gap fill
- OCR quality scoring and retries for tiny UI text
- optional cross-provider OCR fallback (Google -> Azure Read)
- segment refinement to avoid giant transcript blocks
- top-k alignment candidates (not just nearest single frame)

## Pipeline Stages

1. Audio extraction (`ffmpeg`)
2. Frame extraction (scene + periodic + anchors + gap fill)
3. OCR preprocessing
4. Perceptual-hash dedupe with transition preservation
5. Transcription (`faster-whisper` or `AssemblyAI`)
6. OCR (`Tesseract` or `Google Vision`; optional Azure fallback)
7. Alignment (segment -> frame candidates)
8. Windowing
9. Quality report + optional quality gate enforcement

## Installation

```bash
brew install ffmpeg tesseract
python -m pip install -r requirements.txt
```

## Credentials (Optional Cloud Providers)

Set only what you use.

```bash
# AssemblyAI (ASR + diarization)
export ASSEMBLYAI_API_KEY="..."

# Google Vision OCR (ADC recommended)
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"

# Optional Azure OCR fallback
export AZURE_VISION_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_VISION_KEY="..."
```

## Quick Start

### Quality-first (default profile)

```bash
python -m videopipe \
  --video ./input/video.mov \
  --out ./output \
  --profile quality_first \
  --frame-policy hybrid \
  --periodic-interval-seconds 15 \
  --max-frame-gap-seconds 15 \
  --transcribe-provider assemblyai \
  --ocr-provider google \
  --ocr-fallback-provider azure \
  --google-ocr-feature document_text_detection \
  --force-language en \
  --enforce-quality true
```

### Local-safe (offline stack)

```bash
python -m videopipe \
  --video ./input/video.mov \
  --out ./output \
  --profile local_safe \
  --transcribe-provider whisper \
  --ocr-provider tesseract
```

### Smoke test

```bash
python -m videopipe --video ./input/video.mov --max-seconds 60
```

## Optional Interaction Sidecar

Pass `--events-log` to include click/scroll/key events in windows.

Supported formats: `.jsonl`, `.json`, `.csv`, `.tsv`

Minimum event fields:
- timestamp (`ts` or `timestamp`) in seconds from recording start
- event type (`click`, `scroll`, `key`)

Example JSONL:

```json
{"ts": 12.3, "event": "click", "payload": {"x": 811, "y": 412, "button": "left"}}
{"ts": 14.0, "event": "scroll", "payload": {"dy": -380}}
{"ts": 16.2, "event": "key", "payload": {"key": "Enter"}}
```

## Output Layout

For `./input/video.mov`:

```text
./output/video/
  audio.wav
  transcript.json
  transcript_utterances.json
  transcript.srt
  speakers.json
  events.json                  # when --events-log is provided
  frames_raw/
  frames/
  frames_index.json
  kept_frames_index.json
  dropped_frames.json
  frames_ocr.json
  dataset.json
  dataset_windows.json
  quality_report.json
```

## Key Schema Additions

`frames_ocr.json` adds:
- `quality_score`, `quality_flags`, `quality_metrics`
- `fallback_used`, `provider_candidates`

`dataset.json` adds:
- `frame_candidates`, `frame_prev`, `frame_next`
- `attach_score`, `attach_reason`
- `segment_id`, `parent_utterance_id`, `split_reason`

`quality_report.json` includes:
- frame coverage (`max_gap_sec`, `tail_gap_sec`)
- transcript segmentation (`p95_segment_duration_sec`)
- alignment attach rate
- OCR non-empty/low-quality/provider-error metrics
- quality gate pass/fail results

## Important Flags

- `--profile {local_safe,quality_first}`
- `--frame-policy {scene_only,hybrid}`
- `--periodic-interval-seconds 15`
- `--max-frame-gap-seconds 15`
- `--always-include-start-end true`
- `--dedupe-preserve-transitions true`
- `--segment-max-seconds 25`
- `--segment-silence-gap-seconds 0.6`
- `--segment-min-seconds 3`
- `--align-mode {nearest,overlap_topk}`
- `--align-topk 3`
- `--ocr-quality-threshold 0.55`
- `--ocr-fallback-provider {none,azure}`
- `--events-log /path/to/events.jsonl`
- `--enforce-quality true`

## Privacy and Security

- Local mode keeps processing fully offline.
- Cloud mode sends audio/images only to providers you enabled.
- API keys are read from environment variables.
- Do **not** commit keys, `.env` files, or credential JSON files.
- Repo ignore rules already exclude local outputs and secret files.

## Current Scope

Optimized for macOS + Apple Silicon (Python 3.10+), especially screen recordings of web applications where OCR and temporal alignment quality matter.
