# Video-to-LLM (Local First, Cloud Optional)

Convert a screen recording into an LLM-ingestion dataset with:
- timestamped transcript
- scene-change keyframes
- OCR text from keyframes (optimized for tiny UI text)
- alignment between transcript segments and nearest keyframes

Default mode is fully local. Optional paid providers are supported:
- ASR + diarization: AssemblyAI
- OCR: Google Cloud Vision

## Install

```bash
brew install ffmpeg tesseract
python -m pip install -r requirements.txt
```

## Environment (Optional Cloud Providers)

Copy `.env.example` values into your shell environment:

```bash
export ASSEMBLYAI_API_KEY="..."
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
```

`GOOGLE_APPLICATION_CREDENTIALS` can also point to a service-account JSON file, but ADC is recommended for local development.

## Usage

### Local default (offline)

```bash
python -m videopipe \
  --video ./input/video.mov \
  --out ./output \
  --scene-threshold 0.25 \
  --ocr-lang eng \
  --whisper-model large-v3 \
  --force-language en
```

### Cloud quality stack (AssemblyAI + Google OCR)

```bash
python -m videopipe \
  --video ./input/video.mov \
  --out ./output \
  --transcribe-provider assemblyai \
  --assembly-diarization true \
  --assembly-speaker-label-format alpha \
  --ocr-provider google \
  --google-ocr-feature document_text_detection \
  --force-language en
```

### Smoke test (first 60 seconds)

```bash
python -m videopipe --video ./input/video.mov --max-seconds 60
```

## Output Layout

For input `./input/video.mov`, outputs are created under:

```text
./output/video/
  audio.wav
  transcript.json
  transcript.srt
  speakers.json
  frames_raw/
  frames/
  frames_index.json
  kept_frames_index.json
  dropped_frames.json
  frames_ocr.json
  dataset.json
  dataset_windows.json
```

## OCR for Tiny Dashboard Text

Useful defaults:
- `--ocr-scale 2.0` (raise to `2.5` for very tiny text)
- `--ocr-threshold adaptive`
- `--ocr-denoise true`
- `--ocr-sharpen true`
- `--ocr-crops preset` (full + top + left + main + bottom)

Manual crop override:

```bash
python -m videopipe \
  --video ./input/video.mov \
  --ocr-crop "0,0,1600,220" \
  --ocr-crop "0,220,500,1200"
```

## Important CLI Flags

- `--transcribe-provider` (`whisper|assemblyai`, default `whisper`)
- `--ocr-provider` (`tesseract|google`, default `tesseract`)
- `--assembly-diarization` (`true|false`, default `true`)
- `--assembly-speaker-label-format` (`alpha|numeric`, default `alpha`)
- `--google-ocr-feature` (`text_detection|document_text_detection`, default `document_text_detection`)
- `--google-timeout-seconds` (default `30`)
- `--google-max-concurrency` (default `4`)
- `--scene-threshold` (default `0.25`)
- `--align-max-gap` (default `10`)
- `--max-seconds` / `--max-minutes` for smoke tests

## Notes

- Local mode does not send data to external APIs.
- Cloud modes send audio and/or images to selected providers.
- This repository is configured to keep secrets and generated outputs local (`.env*`, credential JSON files, `output*/`, media files).
- Existing output schema is backward compatible; cloud mode adds optional fields:
  - transcript segment: `speaker`, `speaker_confidence`, `asr_provider`
  - frame OCR entry: `ocr_provider`, `provider_meta`
  - dataset record: `seg_speaker`, `asr_provider`, `ocr_provider`
