from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _to_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _normalize_speaker(raw_speaker: Any, mode: str) -> str | None:
    if raw_speaker is None:
        return None
    raw = str(raw_speaker).strip()
    if not raw:
        return None

    if mode == "numeric":
        if raw.isdigit():
            return raw
        if len(raw) == 1 and raw.isalpha():
            return str(ord(raw.upper()) - ord("A") + 1)
        return raw

    if mode != "alpha":
        raise ValueError(f"Unsupported speaker label mode: {mode}")

    if len(raw) == 1 and raw.isalpha():
        return raw.upper()
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx <= 25:
            return chr(ord("A") + idx)
        return f"S{idx}"
    return raw.upper()


def _extract_words(utterance: Any) -> list[dict]:
    out: list[dict] = []
    words = getattr(utterance, "words", None) or []
    for word in words:
        text = str(getattr(word, "text", "")).strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "start": round(_to_seconds(getattr(word, "start", 0)), 3),
                "end": round(_to_seconds(getattr(word, "end", 0)), 3),
                "confidence": getattr(word, "confidence", None),
            }
        )
    return out


def transcribe_audio_assemblyai(
    audio_path: Path,
    language: str = "en",
    diarization: bool = True,
    speaker_label_format: str = "alpha",
) -> tuple[list[dict], list[dict]]:
    try:
        import assemblyai as aai
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("assemblyai is not installed. Run: pip install -r requirements.txt") from exc

    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set. Export your key before using --transcribe-provider assemblyai"
        )

    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speaker_labels=bool(diarization),
        language_code=language,
        speech_models=["universal-2"],
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(str(audio_path), config=config)

    error_msg = getattr(transcript, "error", None)
    if error_msg:
        raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")

    segments: list[dict] = []
    speaker_rows: list[dict] = []

    utterances = list(getattr(transcript, "utterances", None) or [])
    if utterances:
        for idx, utterance in enumerate(utterances, start=1):
            text = str(getattr(utterance, "text", "")).strip()
            if not text:
                continue

            start = round(_to_seconds(getattr(utterance, "start", 0)), 3)
            end = round(_to_seconds(getattr(utterance, "end", 0)), 3)
            confidence = getattr(utterance, "confidence", None)
            speaker = _normalize_speaker(getattr(utterance, "speaker", None), speaker_label_format)
            words = _extract_words(utterance)

            segment = {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
                "speaker_confidence": round(float(confidence), 4) if confidence is not None else None,
                "asr_provider": "assemblyai",
            }
            segments.append(segment)
            speaker_rows.append(
                {
                    "utterance_index": idx,
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                    "confidence": round(float(confidence), 4) if confidence is not None else None,
                    "words": words,
                    "provider": "assemblyai",
                }
            )

    if not segments:
        text = str(getattr(transcript, "text", "")).strip()
        audio_duration = _to_seconds(getattr(transcript, "audio_duration", 0))
        if text:
            segments.append(
                {
                    "start": 0.0,
                    "end": round(audio_duration, 3),
                    "text": text,
                    "speaker": None,
                    "speaker_confidence": None,
                    "asr_provider": "assemblyai",
                }
            )

    return segments, speaker_rows
