from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ALLOWED_EVENT_TYPES = {"click", "scroll", "key"}


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": text}
        return {"raw": text}
    return {"raw": str(payload)}


def _coerce_event_row(item: dict[str, Any], idx: int) -> dict[str, Any] | None:
    raw_ts = item.get("ts")
    if raw_ts is None:
        raw_ts = item.get("timestamp")
    if raw_ts is None:
        raw_ts = item.get("time")
    try:
        ts = float(raw_ts)
    except (TypeError, ValueError):
        return None

    raw_type = str(item.get("event") or item.get("type") or "").strip().lower()
    if raw_type not in ALLOWED_EVENT_TYPES:
        return None

    payload = item.get("payload")
    if payload is None:
        payload = {k: v for k, v in item.items() if k not in {"ts", "timestamp", "time", "event", "type"}}

    return {
        "event_id": idx,
        "timestamp": round(ts, 3),
        "event_type": raw_type,
        "payload": _normalize_payload(payload),
    }


def parse_events_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    suffix = path.suffix.lower()
    events: list[dict[str, Any]] = []

    if suffix in {".jsonl", ".ndjson"}:
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            normalized = _coerce_event_row(obj, idx)
            if normalized:
                events.append(normalized)
        return sorted(events, key=lambda item: item["timestamp"])

    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("events", [])
        if isinstance(data, list):
            for idx, obj in enumerate(data, start=1):
                if not isinstance(obj, dict):
                    continue
                normalized = _coerce_event_row(obj, idx)
                if normalized:
                    events.append(normalized)
        return sorted(events, key=lambda item: item["timestamp"])

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        for idx, row in enumerate(reader, start=1):
            normalized = _coerce_event_row(dict(row), idx)
            if normalized:
                events.append(normalized)
        return sorted(events, key=lambda item: item["timestamp"])

    # fallback: try jsonl first
    for idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        normalized = _coerce_event_row(obj, idx)
        if normalized:
            events.append(normalized)
    return sorted(events, key=lambda item: item["timestamp"])
