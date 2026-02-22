# Video-to-LLM

Convert screen recordings into LLM-ready structured data. One command, zero configuration.

## Quick Start

```bash
brew install ffmpeg tesseract
pip install -e .

videopipe recording.mov ./output
```

That's it. The tool analyzes your video (resolution, scene changes, text density), auto-tunes all parameters, and runs the full pipeline.

## What It Produces

```text
./output/recording/
  audio.wav                    # mono 16kHz audio
  transcript.json              # timed segments (+ speaker diarization)
  transcript.srt               # SubRip subtitles
  frames/                      # preprocessed keyframes
  frames_ocr.json              # OCR text per frame with quality scores
  dataset.json                 # aligned transcript + frame + OCR records
  dataset_windows.json         # windowed chunks for retrieval/analysis
  quality_report.json          # coverage, alignment, OCR quality metrics
```

## How Auto-Tuning Works

Before running the pipeline, `videopipe` probes the video (~30-60s):

1. **Metadata**: resolution, FPS, duration, audio presence
2. **Scene changes**: counts visual transitions to set frame capture rate
3. **Text density**: runs OCR on sample frames to calibrate preprocessing
4. **Provider detection**: checks for cloud API keys, falls back to local tools

Based on these findings, it automatically sets frame intervals, OCR scale factors, deduplication thresholds, and provider selection.

## Cloud Providers (Optional)

Without any API keys, the tool uses local processing (Whisper + Tesseract). For better quality, set cloud provider credentials:

```bash
# AssemblyAI — transcription + speaker diarization
export ASSEMBLYAI_API_KEY="..."

# Google Vision — OCR (ADC recommended)
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"

# Azure Vision — optional OCR fallback
export AZURE_VISION_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_VISION_KEY="..."
```

The tool auto-detects which keys are available and uses the best stack.

## Advanced Usage

For full manual control over 80+ parameters:

```bash
python -m videopipe \
  --video recording.mov \
  --out ./output \
  --profile quality_first \
  --frame-policy hybrid \
  --periodic-interval-seconds 10 \
  --transcribe-provider assemblyai \
  --ocr-provider google \
  --ocr-fallback-provider azure
```

See `python -m videopipe --help` for all options.

## Pipeline Stages

1. Audio extraction (`ffmpeg`)
2. Frame extraction (scene changes + periodic sampling + anchors + gap fill)
3. OCR preprocessing (upscaling, thresholding, denoising)
4. Perceptual-hash deduplication with transition preservation
5. Transcription (`faster-whisper` or `AssemblyAI`)
6. OCR (`Tesseract` or `Google Vision`; optional Azure fallback)
7. Transcript-to-frame alignment (top-k candidates)
8. Windowed dataset chunking
9. Quality report with gate checks

## Why This Exists

Naive "sample every N seconds" pipelines miss scene changes, produce blurry OCR on dense UI text, and create giant transcript blocks. This tool uses adaptive frame capture, OCR quality scoring with provider fallback, segment refinement, and multi-candidate alignment to produce structured data that LLMs can actually use.

## Privacy

- Without cloud API keys, all processing stays local.
- Cloud mode sends audio/images only to providers you enabled.
- API keys are read from environment variables only.

## Requirements

- Python 3.10+
- macOS (optimized for Apple Silicon) or Linux
- `ffmpeg` and `tesseract` (system binaries)

## License

MIT
