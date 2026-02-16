from __future__ import annotations

from typing import Iterable


def _sorted_unique_timestamps(timestamps: Iterable[float], epsilon: float = 0.2) -> list[float]:
    values = sorted(float(ts) for ts in timestamps)
    out: list[float] = []
    for ts in values:
        if not out or abs(ts - out[-1]) > epsilon:
            out.append(ts)
    return out


def required_anchor_timestamps(
    entries: list[dict],
    *,
    duration_seconds: float,
    always_include_start_end: bool,
    epsilon: float = 0.2,
) -> list[float]:
    if not always_include_start_end:
        return []
    if duration_seconds <= 0:
        return []

    existing = _sorted_unique_timestamps((float(item["timestamp"]) for item in entries), epsilon=epsilon)
    anchors = [0.0, max(0.0, float(duration_seconds) - 0.5)]
    missing: list[float] = []
    for anchor in anchors:
        if not any(abs(anchor - ts) <= epsilon for ts in existing):
            missing.append(round(anchor, 3))
    return missing


def merge_entries_by_timestamp(entries: list[dict], *, epsilon: float = 0.2) -> list[dict]:
    merged: list[dict] = []
    for item in sorted(entries, key=lambda rec: float(rec["timestamp"])):
        ts = float(item["timestamp"])
        if merged and abs(ts - float(merged[-1]["timestamp"])) <= epsilon:
            continue
        merged.append(item)
    for idx, item in enumerate(merged, start=1):
        item["original_index"] = idx
    return merged


def compute_gap_fill_timestamps(
    timestamps: list[float],
    *,
    duration_seconds: float,
    max_gap_seconds: float,
    epsilon: float = 0.2,
    always_include_start_end: bool = True,
) -> list[float]:
    if max_gap_seconds <= 0:
        return []

    points = [float(ts) for ts in timestamps]
    if always_include_start_end and duration_seconds > 0:
        points.append(0.0)
        points.append(max(0.0, float(duration_seconds) - 0.5))
    ordered = _sorted_unique_timestamps(points, epsilon=epsilon)
    if not ordered:
        return [0.0] if duration_seconds > 0 else []

    fill: list[float] = []
    prev = ordered[0]
    for current in ordered[1:]:
        gap = current - prev
        if gap > max_gap_seconds:
            ts = prev + max_gap_seconds
            while ts < current - epsilon:
                fill.append(round(ts, 3))
                ts += max_gap_seconds
        prev = current
    return _sorted_unique_timestamps(fill, epsilon=epsilon)
