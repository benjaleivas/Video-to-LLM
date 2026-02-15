from __future__ import annotations

from bisect import bisect_left


def _nearest_frame(
    target_ts: float,
    frame_timestamps: list[float],
    frame_records: list[dict],
) -> dict | None:
    if not frame_timestamps:
        return None
    idx = bisect_left(frame_timestamps, target_ts)
    candidates: list[dict] = []
    if idx < len(frame_records):
        candidates.append(frame_records[idx])
    if idx > 0:
        candidates.append(frame_records[idx - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda rec: abs(float(rec["timestamp"]) - target_ts))


def build_dataset(transcript_segments: list[dict], frame_ocr_records: list[dict], align_max_gap: float = 10.0) -> list[dict]:
    frames_sorted = sorted(frame_ocr_records, key=lambda item: item["timestamp"])
    frame_timestamps = [float(item["timestamp"]) for item in frames_sorted]

    dataset: list[dict] = []
    for seg in transcript_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        mid = (start + end) / 2.0
        nearest = _nearest_frame(mid, frame_timestamps, frames_sorted)

        attach = None
        if nearest is not None:
            gap = abs(float(nearest["timestamp"]) - mid)
            if gap <= align_max_gap:
                attach = nearest

        dataset.append(
            {
                "seg_start": start,
                "seg_end": end,
                "seg_text": seg["text"],
                "seg_speaker": seg.get("speaker"),
                "speaker_confidence": seg.get("speaker_confidence"),
                "asr_provider": seg.get("asr_provider"),
                "frame_timestamp": attach["timestamp"] if attach else None,
                "frame_path": attach["frame_path"] if attach else None,
                "frame_ocr_text": attach["ocr"]["full_text"] if attach else None,
                "frame_ocr_crops": attach["ocr"]["crops"] if attach else None,
                "ocr_provider": attach.get("ocr_provider") if attach else None,
            }
        )
    return dataset


def _key_frame(frames: list[dict]) -> dict | None:
    if not frames:
        return None

    def score(frame: dict) -> tuple[int, float]:
        text = frame.get("ocr", {}).get("full_text", "")
        avg_conf = frame.get("ocr", {}).get("avg_conf")
        return (len(text.strip()), float(avg_conf) if avg_conf is not None else -1.0)

    best = max(frames, key=score)
    return {
        "timestamp": best["timestamp"],
        "frame_path": best["frame_path"],
        "ocr_text": best["ocr"]["full_text"],
        "avg_conf": best["ocr"]["avg_conf"],
        "ocr_provider": best.get("ocr_provider"),
    }


def build_dataset_windows(
    transcript_segments: list[dict],
    frame_ocr_records: list[dict],
    window_seconds: float = 45.0,
) -> list[dict]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    max_seg_time = max((float(seg["end"]) for seg in transcript_segments), default=0.0)
    max_frame_time = max((float(frame["timestamp"]) for frame in frame_ocr_records), default=0.0)
    max_time = max(max_seg_time, max_frame_time)
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
        if segs or frames:
            transcript_text = " ".join(seg["text"].strip() for seg in segs if seg["text"].strip()).strip()
            frame_records = [
                {
                    "timestamp": frame["timestamp"],
                    "frame_path": frame["frame_path"],
                    "ocr_text": frame["ocr"]["full_text"],
                    "avg_conf": frame["ocr"]["avg_conf"],
                    "ocr_provider": frame.get("ocr_provider"),
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
