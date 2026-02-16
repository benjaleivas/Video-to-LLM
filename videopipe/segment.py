from __future__ import annotations

import re
from typing import Any


PUNCT_END_RE = re.compile(r"[.!?;:]$")


def _join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tok.strip() for tok in tokens if tok and tok.strip()).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def _normalize_utterance_row(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "utterance_id": idx,
        "start": round(float(raw.get("start", 0.0)), 3),
        "end": round(float(raw.get("end", 0.0)), 3),
        "text": str(raw.get("text") or "").strip(),
        "speaker": raw.get("speaker"),
        "speaker_confidence": raw.get("confidence"),
        "asr_provider": "assemblyai",
        "words": raw.get("words") or [],
    }


def _split_utterance_words(
    utterance: dict[str, Any],
    *,
    max_seconds: float,
    silence_gap_seconds: float,
) -> list[dict[str, Any]]:
    words = utterance.get("words") or []
    if not words:
        return [
            {
                "start": float(utterance["start"]),
                "end": float(utterance["end"]),
                "text": str(utterance.get("text") or "").strip(),
                "speaker": utterance.get("speaker"),
                "speaker_confidence": utterance.get("speaker_confidence"),
                "parent_utterance_id": utterance["utterance_id"],
                "split_reason": "utterance_fallback",
            }
        ]

    chunks: list[dict[str, Any]] = []
    chunk_tokens: list[str] = []
    chunk_start = float(words[0].get("start", utterance["start"]))
    last_end = chunk_start
    split_reason = "punctuation"

    for i, word in enumerate(words):
        token = str(word.get("text") or "").strip()
        if not token:
            continue
        start = float(word.get("start", last_end))
        end = float(word.get("end", start))
        gap = max(0.0, start - last_end)
        duration_if_add = end - chunk_start

        force_split = False
        reason = None
        if chunk_tokens and gap >= silence_gap_seconds:
            force_split = True
            reason = "silence_gap"
        elif chunk_tokens and duration_if_add >= max_seconds:
            force_split = True
            reason = "max_duration"

        if force_split:
            text = _join_tokens(chunk_tokens)
            if text:
                chunks.append(
                    {
                        "start": round(chunk_start, 3),
                        "end": round(last_end, 3),
                        "text": text,
                        "speaker": utterance.get("speaker"),
                        "speaker_confidence": utterance.get("speaker_confidence"),
                        "parent_utterance_id": utterance["utterance_id"],
                        "split_reason": reason or split_reason,
                    }
                )
            chunk_tokens = []
            chunk_start = start

        chunk_tokens.append(token)
        last_end = end

        if PUNCT_END_RE.search(token):
            text = _join_tokens(chunk_tokens)
            if text:
                chunks.append(
                    {
                        "start": round(chunk_start, 3),
                        "end": round(last_end, 3),
                        "text": text,
                        "speaker": utterance.get("speaker"),
                        "speaker_confidence": utterance.get("speaker_confidence"),
                        "parent_utterance_id": utterance["utterance_id"],
                        "split_reason": "punctuation",
                    }
                )
            chunk_tokens = []
            if i + 1 < len(words):
                next_word = words[i + 1]
                chunk_start = float(next_word.get("start", last_end))
            else:
                chunk_start = last_end

    if chunk_tokens:
        text = _join_tokens(chunk_tokens)
        if text:
            chunks.append(
                {
                    "start": round(chunk_start, 3),
                    "end": round(last_end, 3),
                    "text": text,
                    "speaker": utterance.get("speaker"),
                    "speaker_confidence": utterance.get("speaker_confidence"),
                    "parent_utterance_id": utterance["utterance_id"],
                    "split_reason": "tail",
                }
            )

    if not chunks:
        chunks.append(
            {
                "start": round(float(utterance["start"]), 3),
                "end": round(float(utterance["end"]), 3),
                "text": str(utterance.get("text") or "").strip(),
                "speaker": utterance.get("speaker"),
                "speaker_confidence": utterance.get("speaker_confidence"),
                "parent_utterance_id": utterance["utterance_id"],
                "split_reason": "empty_words_fallback",
            }
        )
    return chunks


def _merge_short_segments(segments: list[dict[str, Any]], min_seconds: float) -> list[dict[str, Any]]:
    if min_seconds <= 0 or len(segments) <= 1:
        return segments

    merged: list[dict[str, Any]] = []
    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        if duration >= min_seconds or not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        if prev.get("speaker") == seg.get("speaker"):
            combined_text = _join_tokens([prev.get("text", ""), seg.get("text", "")])
            prev["text"] = combined_text
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
            prev["split_reason"] = f"{prev.get('split_reason','unknown')}+merged_short"
        else:
            merged.append(seg)
    return merged


def refine_segments_from_utterances(
    utterances: list[dict[str, Any]],
    *,
    max_seconds: float = 25.0,
    silence_gap_seconds: float = 0.6,
    min_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_utterances = [_normalize_utterance_row(item, idx=i) for i, item in enumerate(utterances, start=1)]

    out: list[dict[str, Any]] = []
    for row in normalized_utterances:
        out.extend(
            _split_utterance_words(
                row,
                max_seconds=max_seconds,
                silence_gap_seconds=silence_gap_seconds,
            )
        )

    out = sorted(out, key=lambda item: (float(item["start"]), float(item["end"])))
    out = _merge_short_segments(out, min_seconds=min_seconds)

    for idx, seg in enumerate(out, start=1):
        seg["segment_id"] = idx
        seg["start"] = round(float(seg["start"]), 3)
        seg["end"] = round(float(seg["end"]), 3)
        seg["asr_provider"] = "assemblyai"
        if seg.get("speaker_confidence") is not None:
            try:
                seg["speaker_confidence"] = round(float(seg["speaker_confidence"]), 4)
            except (TypeError, ValueError):
                seg["speaker_confidence"] = None

    for row in normalized_utterances:
        row["speaker_confidence"] = row.pop("speaker_confidence", None)
    return out, normalized_utterances
