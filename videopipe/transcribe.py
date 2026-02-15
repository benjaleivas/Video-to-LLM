from __future__ import annotations

from pathlib import Path

from .utils import format_srt_timestamp


def transcribe_audio(
    audio_path: Path,
    model_name: str = "large-v3",
    language: str = "en",
    compute_type: str = "int8",
) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    segments_iter, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )
    segments: list[dict] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
                "asr_provider": "whisper",
            }
        )
    return segments


def segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(float(seg["start"]))
        end_ts = format_srt_timestamp(float(seg["end"]))
        text = str(seg["text"]).strip()
        lines.extend([str(i), f"{start_ts} --> {end_ts}", text, ""])
    return "\n".join(lines).strip() + "\n"
