from __future__ import annotations

from bisect import bisect_left
from typing import Any


def _frame_brief(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": frame.get("timestamp"),
        "frame_path": frame.get("frame_path"),
        "ocr_text": (frame.get("ocr") or {}).get("full_text"),
        "avg_conf": (frame.get("ocr") or {}).get("avg_conf"),
        "ocr_provider": frame.get("ocr_provider"),
        "quality_score": frame.get("quality_score"),
    }


def _nearest_prev_next(target_ts: float, frame_timestamps: list[float], frame_records: list[dict]) -> tuple[dict | None, dict | None]:
    if not frame_timestamps:
        return None, None
    idx = bisect_left(frame_timestamps, target_ts)
    prev_rec = frame_records[idx - 1] if idx > 0 else None
    next_rec = frame_records[idx] if idx < len(frame_records) else None
    return prev_rec, next_rec


def _overlap_distance(seg_start: float, seg_end: float, frame_ts: float) -> float:
    if seg_start <= frame_ts <= seg_end:
        return 0.0
    if frame_ts < seg_start:
        return seg_start - frame_ts
    return frame_ts - seg_end


def _score_candidate(
    *,
    seg_start: float,
    seg_end: float,
    frame: dict,
    align_max_gap: float,
) -> float:
    ts = float(frame["timestamp"])
    seg_mid = (seg_start + seg_end) / 2.0
    proximity = 1.0 - min(1.0, abs(ts - seg_mid) / max(align_max_gap, 1e-6))
    overlap_distance = _overlap_distance(seg_start, seg_end, ts)
    overlap_score = 1.0 - min(1.0, overlap_distance / max(align_max_gap, 1e-6))
    quality = float(frame.get("quality_score") or 0.0)
    return (overlap_score * 0.5) + (proximity * 0.3) + (quality * 0.2)


def _candidate_frames(
    *,
    seg_start: float,
    seg_end: float,
    frames_sorted: list[dict],
    align_max_gap: float,
    align_mode: str,
) -> list[dict]:
    if not frames_sorted:
        return []

    if align_mode == "nearest":
        mid = (seg_start + seg_end) / 2.0
        nearest = min(frames_sorted, key=lambda rec: abs(float(rec["timestamp"]) - mid))
        gap = abs(float(nearest["timestamp"]) - mid)
        if gap <= align_max_gap:
            nearest = {**nearest, "_attach_score": 1.0 - min(1.0, gap / max(align_max_gap, 1e-6))}
            return [nearest]
        return []

    # overlap_topk mode
    start_bound = seg_start - align_max_gap
    end_bound = seg_end + align_max_gap
    in_range = [
        frame
        for frame in frames_sorted
        if start_bound <= float(frame["timestamp"]) <= end_bound
    ]
    scored = []
    for frame in in_range:
        score = _score_candidate(seg_start=seg_start, seg_end=seg_end, frame=frame, align_max_gap=align_max_gap)
        scored.append({**frame, "_attach_score": score})
    scored.sort(key=lambda item: (float(item["_attach_score"]), float(item.get("quality_score") or 0.0)), reverse=True)
    return scored


def build_dataset(
    transcript_segments: list[dict],
    frame_ocr_records: list[dict],
    align_max_gap: float = 10.0,
    *,
    align_topk: int = 3,
    align_mode: str = "overlap_topk",
) -> list[dict]:
    if align_mode not in {"nearest", "overlap_topk"}:
        raise ValueError("align_mode must be one of: nearest, overlap_topk")

    frames_sorted = sorted(frame_ocr_records, key=lambda item: item["timestamp"])
    frame_timestamps = [float(item["timestamp"]) for item in frames_sorted]

    dataset: list[dict] = []
    for seg in transcript_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        mid = (start + end) / 2.0

        candidates = _candidate_frames(
            seg_start=start,
            seg_end=end,
            frames_sorted=frames_sorted,
            align_max_gap=align_max_gap,
            align_mode=align_mode,
        )
        top_candidates = candidates[: max(1, int(align_topk))]
        attach = top_candidates[0] if top_candidates else None

        prev_frame, next_frame = _nearest_prev_next(mid, frame_timestamps, frames_sorted)

        if attach is not None:
            attach_reason = "candidate_attached"
            attach_score = float(attach.get("_attach_score") or 0.0)
        else:
            attach_reason = "no_in_range_candidate"
            attach_score = 0.0

        dataset.append(
            {
                "seg_start": start,
                "seg_end": end,
                "seg_text": seg["text"],
                "seg_speaker": seg.get("speaker"),
                "speaker_confidence": seg.get("speaker_confidence"),
                "asr_provider": seg.get("asr_provider"),
                "segment_id": seg.get("segment_id"),
                "parent_utterance_id": seg.get("parent_utterance_id"),
                "split_reason": seg.get("split_reason"),
                "frame_timestamp": attach["timestamp"] if attach else None,
                "frame_path": attach["frame_path"] if attach else None,
                "frame_ocr_text": attach["ocr"]["full_text"] if attach else None,
                "frame_ocr_crops": attach["ocr"]["crops"] if attach else None,
                "ocr_provider": attach.get("ocr_provider") if attach else None,
                "frame_candidates": [
                    {
                        "timestamp": cand.get("timestamp"),
                        "frame_path": cand.get("frame_path"),
                        "ocr_provider": cand.get("ocr_provider"),
                        "quality_score": cand.get("quality_score"),
                        "attach_score": round(float(cand.get("_attach_score") or 0.0), 4),
                    }
                    for cand in top_candidates
                ],
                "frame_prev": _frame_brief(prev_frame) if prev_frame else None,
                "frame_next": _frame_brief(next_frame) if next_frame else None,
                "attach_score": round(float(attach_score), 4),
                "attach_reason": attach_reason,
            }
        )
    return dataset


def _key_frame(frames: list[dict]) -> dict | None:
    if not frames:
        return None

    def score(frame: dict) -> tuple[float, int, float]:
        text = frame.get("ocr", {}).get("full_text", "")
        avg_conf = frame.get("ocr", {}).get("avg_conf")
        quality = float(frame.get("quality_score") or 0.0)
        return (quality, len(text.strip()), float(avg_conf) if avg_conf is not None else -1.0)

    best = max(frames, key=score)
    return {
        "timestamp": best["timestamp"],
        "frame_path": best["frame_path"],
        "ocr_text": best["ocr"]["full_text"],
        "avg_conf": best["ocr"]["avg_conf"],
        "ocr_provider": best.get("ocr_provider"),
        "quality_score": best.get("quality_score"),
    }


def build_dataset_windows(
    transcript_segments: list[dict],
    frame_ocr_records: list[dict],
    window_seconds: float = 45.0,
    *,
    events: list[dict] | None = None,
) -> list[dict]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    events = events or []

    max_seg_time = max((float(seg["end"]) for seg in transcript_segments), default=0.0)
    max_frame_time = max((float(frame["timestamp"]) for frame in frame_ocr_records), default=0.0)
    max_event_time = max((float(event["timestamp"]) for event in events), default=0.0)
    max_time = max(max_seg_time, max_frame_time, max_event_time)
    if max_time <= 0:
        return []

    windows: list[dict] = []
    window_start = 0.0
    while window_start <= max_time:
        window_end = window_start + window_seconds
        segs = [
            seg
            for seg in transcript_segments
            if float(seg["end"]) > window_start and float(seg["start"]) < window_end
        ]
        frames = [
            frame
            for frame in frame_ocr_records
            if window_start <= float(frame["timestamp"]) < window_end
        ]
        window_events = [
            event
            for event in events
            if window_start <= float(event["timestamp"]) < window_end
        ]
        if segs or frames or window_events:
            transcript_text = " ".join(seg["text"].strip() for seg in segs if seg["text"].strip()).strip()
            frame_records = [
                {
                    "timestamp": frame["timestamp"],
                    "frame_path": frame["frame_path"],
                    "ocr_text": frame["ocr"]["full_text"],
                    "avg_conf": frame["ocr"]["avg_conf"],
                    "ocr_provider": frame.get("ocr_provider"),
                    "quality_score": frame.get("quality_score"),
                }
                for frame in frames
            ]
            speakers_present = sorted({seg.get("speaker") for seg in segs if seg.get("speaker")})
            asr_providers = sorted({seg.get("asr_provider") for seg in segs if seg.get("asr_provider")})
            ocr_providers = sorted({frame.get("ocr_provider") for frame in frames if frame.get("ocr_provider")})

            windows.append(
                {
                    "window_start": round(window_start, 3),
                    "window_end": round(window_end, 3),
                    "transcript_text": transcript_text,
                    "segments": segs,
                    "frames": frame_records,
                    "events": window_events,
                    "key_frame": _key_frame(frames),
                    "speakers_present": speakers_present,
                    "provider_summary": {
                        "asr_providers": asr_providers,
                        "ocr_providers": ocr_providers,
                    },
                }
            )
        window_start = window_end
    return windows
