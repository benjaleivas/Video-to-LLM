# Video-to-LLM

LLMs can't watch videos. You can't drag a screen recording into ChatGPT and ask "what happened at minute 3?" — the model has no way to process video frames, read on-screen text, or follow a spoken narrative across time.

**videopipe** bridges that gap. It converts any video into a structured directory of transcripts, keyframe images, and OCR text — all timestamp-aligned and chunked into windows that fit directly into an LLM context.

```bash
brew install ffmpeg tesseract
pip install -e .

videopipe recording.mov ./output
```

One command, zero configuration. The tool probes your video, auto-tunes every parameter, and runs the full pipeline.

## What You Get

```text
./output/recording/
  dataset_windows.json    ← the LLM-ready file
  dataset.json            aligned transcript + frame + OCR records
  transcript.json         timed transcript segments with speakers
  transcript.srt          SubRip subtitles
  frames/                 preprocessed keyframes (PNG)
  frames_ocr.json         OCR text per frame with quality scores
  quality_report.json     coverage, alignment, OCR quality metrics
  audio.wav               mono 16kHz extracted audio
  videopipe.log           full processing log
  README.md               run summary and reproduction command
```

### The Key File: `dataset_windows.json`

This is what you feed to an LLM. Each window is a self-contained chunk of the video covering a time range (default 45 seconds):

```json
{
  "window_index": 0,
  "time_range": [0.0, 45.0],
  "transcript_segments": [
    {"start": 2.1, "end": 8.4, "text": "Let me show you the dashboard...", "speaker": "A"}
  ],
  "frames": [
    {"timestamp": 3.5, "frame_path": "frames/frame_00m03s500.png", "ocr_text": "Revenue: $142K"}
  ]
}
```

You can pass one or more windows to a model alongside the referenced images, and the model can answer questions about what was shown, said, and displayed on screen during that time range.

## How It Works

Before running the pipeline, `videopipe` probes the video (~30-60s):

1. **Metadata** — resolution, FPS, duration, audio presence
2. **Scene changes** — counts visual transitions to set frame capture rate
3. **Text density** — runs OCR on sample frames to calibrate preprocessing
4. **Provider detection** — validates cloud API credentials with real pings, falls back to local tools

Based on these findings, it auto-tunes frame intervals, OCR scale factors, deduplication thresholds, and provider selection. You see a comparison table of defaults vs. tuned values before the pipeline runs.

## Cloud Providers (Optional)

Without any API keys, everything runs locally using Whisper (transcription) and Tesseract (OCR). For better accuracy, set cloud credentials in your environment:

```bash
# .env or shell profile

# AssemblyAI — transcription with speaker diarization
ASSEMBLYAI_API_KEY="..."

# Google Vision — OCR (Application Default Credentials recommended)
GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"

# Azure Vision — optional OCR fallback when Google returns low-confidence results
AZURE_VISION_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
AZURE_VISION_KEY="..."
```

The tool auto-detects which keys are available, validates them with lightweight API pings, and uses the best available stack. If a cloud provider fails validation, it falls back to local processing with a warning.

## Advanced Usage

For full manual control:

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

Run `python -m videopipe --help` for all options.

## Pipeline Stages

1. **Audio extraction** — `ffmpeg` demux to mono 16kHz WAV
2. **Frame extraction** — scene-change detection + periodic sampling + anchor frames + gap fill
3. **Preprocessing** — upscaling, adaptive thresholding, denoising, sharpening
4. **Deduplication** — perceptual hashing with transition preservation
5. **Transcription** — `faster-whisper` (local) or AssemblyAI (cloud) with speaker diarization
6. **OCR** — Tesseract (local) or Google Vision (cloud), with optional Azure fallback
7. **Alignment** — maps transcript segments to their best-matching keyframes
8. **Windowing** — chunks the aligned dataset into time-based windows for LLM consumption
9. **Quality report** — coverage checks, alignment scores, OCR quality metrics with gate checks

## Privacy

- Without cloud API keys, all processing stays on your machine.
- Cloud mode sends audio/images only to providers you explicitly enabled via environment variables.
- No data is stored by this tool beyond the output directory.

## Requirements

- Python 3.10+
- macOS (optimized for Apple Silicon) or Linux
- System binaries: `ffmpeg` and `tesseract`

## License

MIT
