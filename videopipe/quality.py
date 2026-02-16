from __future__ import annotations

import math
import re
from typing import Any


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
TOKEN_RE = re.compile(r"\S+")


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_conf_to_unit(avg_conf: float | None) -> float:
    if avg_conf is None:
        return 0.5
    return _clamp01(float(avg_conf) / 100.0)


def _lexical_metrics(text: str) -> dict[str, float]:
    tokens = TOKEN_RE.findall(text)
    words = WORD_RE.findall(text)
    token_count = len(tokens)
    word_count = len(words)

    if token_count == 0:
        return {
            "token_count": 0.0,
            "word_count": 0.0,
            "lexicality": 0.0,
            "weird_token_ratio": 1.0,
            "long_joined_ratio": 0.0,
            "structure_score": 0.0,
        }

    weird = 0
    long_joined = 0
    for tok in tokens:
        alpha_count = sum(1 for ch in tok if ch.isalpha())
        if alpha_count == 0:
            weird += 1
            continue
        bad_chars = sum(1 for ch in tok if not (ch.isalpha() or ch.isdigit() or ch in {"_", "-", "'", "."}))
        if bad_chars / max(1, len(tok)) > 0.25:
            weird += 1
        if len(tok) >= 18 and alpha_count / max(1, len(tok)) > 0.8 and "-" not in tok and "_" not in tok:
            long_joined += 1

    lexicality = word_count / token_count
    weird_token_ratio = weird / token_count
    long_joined_ratio = long_joined / token_count

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    newline_density = min(1.0, len(lines) / 16.0)
    punctuation_density = min(1.0, sum(1 for ch in text if ch in ".,:;!?()[]") / max(1, len(text) / 24.0))
    structure_score = _clamp01((newline_density * 0.45) + (punctuation_density * 0.55))

    return {
        "token_count": float(token_count),
        "word_count": float(word_count),
        "lexicality": _clamp01(lexicality),
        "weird_token_ratio": _clamp01(weird_token_ratio),
        "long_joined_ratio": _clamp01(long_joined_ratio),
        "structure_score": structure_score,
    }


def score_ocr_text(text: str, avg_conf: float | None) -> dict[str, Any]:
    compact = " ".join((text or "").split()).strip()
    metrics = _lexical_metrics(text or "")
    conf_score = _safe_conf_to_unit(avg_conf)

    lexical_score = _clamp01(
        (metrics["lexicality"] * 0.65)
        + ((1.0 - metrics["weird_token_ratio"]) * 0.25)
        + ((1.0 - metrics["long_joined_ratio"]) * 0.10)
    )
    structural_score = metrics["structure_score"]

    if not compact:
        quality_score = 0.0
    else:
        quality_score = _clamp01((conf_score * 0.45) + (lexical_score * 0.40) + (structural_score * 0.15))

    flags: list[str] = []
    if conf_score < 0.65:
        flags.append("low_confidence")
    if lexical_score < 0.60:
        flags.append("low_lexicality")
    if metrics["long_joined_ratio"] > 0.08:
        flags.append("suspicious_joined_tokens")
    if quality_score < 0.55:
        flags.append("garble_like")

    return {
        "quality_score": round(float(quality_score), 4),
        "quality_flags": flags,
        "metrics": {
            "text_len": len(compact),
            "token_count": int(metrics["token_count"]),
            "word_count": int(metrics["word_count"]),
            "lexicality": round(float(metrics["lexicality"]), 4),
            "weird_token_ratio": round(float(metrics["weird_token_ratio"]), 4),
            "long_joined_ratio": round(float(metrics["long_joined_ratio"]), 4),
            "structure_score": round(float(metrics["structure_score"]), 4),
            "confidence_score": round(float(conf_score), 4),
            "lexical_score": round(float(lexical_score), 4),
            "structural_score": round(float(structural_score), 4),
        },
    }


def summarize_quality_report(
    *,
    duration_seconds: float,
    kept_frames: list[dict],
    transcript_segments: list[dict],
    dataset_records: list[dict],
    frame_ocr_records: list[dict],
    window_seconds: float,
) -> dict[str, Any]:
    frame_timestamps = sorted(float(item["timestamp"]) for item in kept_frames)
    max_gap = 0.0
    if frame_timestamps:
        prev = 0.0
        for ts in frame_timestamps:
            max_gap = max(max_gap, ts - prev)
            prev = ts
        tail_gap = max(0.0, duration_seconds - frame_timestamps[-1])
        max_gap = max(max_gap, tail_gap)
    else:
        tail_gap = duration_seconds
        max_gap = duration_seconds

    seg_durations = sorted(float(seg["end"]) - float(seg["start"]) for seg in transcript_segments)
    if seg_durations:
        idx = min(len(seg_durations) - 1, int(math.floor(0.95 * (len(seg_durations) - 1))))
        p95_seg = seg_durations[idx]
    else:
        p95_seg = 0.0

    dataset_total = len(dataset_records)
    attached = sum(1 for rec in dataset_records if rec.get("frame_path"))
    attach_rate = (attached / dataset_total) if dataset_total else 0.0

    ocr_total = len(frame_ocr_records)
    nonempty = 0
    low_quality = 0
    provider_errors = 0
    fallback_used = 0
    retry_invocations = 0
    azure_candidate_count = 0
    for rec in frame_ocr_records:
        ocr = rec.get("ocr") or {}
        if str(ocr.get("full_text") or "").strip():
            nonempty += 1
        if float(rec.get("quality_score") or 0.0) < 0.55:
            low_quality += 1
        if (rec.get("provider_meta") or {}).get("errors"):
            provider_errors += 1
        if rec.get("fallback_used"):
            fallback_used += 1
        candidates = rec.get("provider_candidates") or []
        retry_invocations += max(0, len(candidates) - 1)
        azure_candidate_count += sum(1 for cand in candidates if cand.get("provider") == "azure")

    ocr_nonempty_rate = (nonempty / ocr_total) if ocr_total else 0.0
    low_quality_rate = (low_quality / ocr_total) if ocr_total else 0.0

    return {
        "version": 1,
        "duration_seconds": round(float(duration_seconds), 3),
        "window_seconds": round(float(window_seconds), 3),
        "coverage": {
            "kept_frames": len(kept_frames),
            "max_gap_sec": round(float(max_gap), 3),
            "tail_gap_sec": round(float(tail_gap), 3),
        },
        "transcript": {
            "segment_count": len(transcript_segments),
            "p95_segment_duration_sec": round(float(p95_seg), 3),
        },
        "alignment": {
            "dataset_records": dataset_total,
            "attached_records": attached,
            "attach_rate": round(float(attach_rate), 4),
        },
        "ocr": {
            "frame_records": ocr_total,
            "nonempty_rate": round(float(ocr_nonempty_rate), 4),
            "low_quality_rate": round(float(low_quality_rate), 4),
            "provider_error_count": provider_errors,
            "fallback_used_count": fallback_used,
        },
        "ocr_retry_stats": {
            "retry_invocations": retry_invocations,
            "azure_candidate_count": azure_candidate_count,
            "fallback_used_count": fallback_used,
        },
        "quality_gates": {
            "min_attach_rate": 0.85,
            "max_p95_segment_duration_sec": 30.0,
            "max_gap_sec": 15.0,
            "max_tail_gap_sec": 15.0,
            "min_ocr_nonempty_rate": 0.98,
            "max_low_quality_rate": 0.15,
            "max_provider_error_count": 0,
        },
    }


def evaluate_quality_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    gates = report.get("quality_gates") or {}
    coverage = report.get("coverage") or {}
    transcript = report.get("transcript") or {}
    alignment = report.get("alignment") or {}
    ocr = report.get("ocr") or {}

    checks = [
        ("attach_rate", float(alignment.get("attach_rate") or 0.0), ">=", float(gates.get("min_attach_rate") or 0.85)),
        (
            "p95_segment_duration_sec",
            float(transcript.get("p95_segment_duration_sec") or 0.0),
            "<=",
            float(gates.get("max_p95_segment_duration_sec") or 30.0),
        ),
        ("max_gap_sec", float(coverage.get("max_gap_sec") or 0.0), "<=", float(gates.get("max_gap_sec") or 15.0)),
        ("tail_gap_sec", float(coverage.get("tail_gap_sec") or 0.0), "<=", float(gates.get("max_tail_gap_sec") or 15.0)),
        (
            "ocr_nonempty_rate",
            float(ocr.get("nonempty_rate") or 0.0),
            ">=",
            float(gates.get("min_ocr_nonempty_rate") or 0.98),
        ),
        (
            "low_quality_rate",
            float(ocr.get("low_quality_rate") or 0.0),
            "<=",
            float(gates.get("max_low_quality_rate") or 0.15),
        ),
        (
            "provider_error_count",
            float(ocr.get("provider_error_count") or 0.0),
            "<=",
            float(gates.get("max_provider_error_count") or 0.0),
        ),
    ]

    results: list[dict[str, Any]] = []
    for name, observed, op, target in checks:
        if op == ">=":
            ok = observed >= target
        else:
            ok = observed <= target
        results.append(
            {
                "name": name,
                "observed": round(float(observed), 4),
                "operator": op,
                "target": round(float(target), 4),
                "passed": bool(ok),
            }
        )
    return results
