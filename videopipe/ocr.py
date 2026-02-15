from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Iterable

import imagehash
from PIL import Image

from .utils import log


@dataclass
class OcrOptions:
    lang: str = "eng"
    psm: int = 6
    second_psm: int | None = None
    oem: int = 1
    crops_mode: str = "preset"  # none | preset
    manual_crops: list[tuple[int, int, int, int]] = field(default_factory=list)


def parse_manual_crops(raw_crops: Iterable[str] | None) -> list[tuple[int, int, int, int]]:
    if not raw_crops:
        return []
    parsed: list[tuple[int, int, int, int]] = []
    for raw in raw_crops:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid --ocr-crop value {raw!r}. Expected x1,y1,x2,y2.")
        try:
            x1, y1, x2, y2 = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid --ocr-crop value {raw!r}. Coordinates must be integers.") from exc
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid --ocr-crop value {raw!r}. Require x2>x1 and y2>y1.")
        parsed.append((x1, y1, x2, y2))
    return parsed


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_crop_boxes(
    width: int,
    height: int,
    crops_mode: str,
    manual_crops: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = [("FULL", (0, 0, width, height))]
    if crops_mode == "preset":
        boxes.extend(
            [
                ("TOP", (0, 0, width, int(height * 0.20))),
                ("LEFT", (0, 0, int(width * 0.25), height)),
                (
                    "MAIN",
                    (
                        int(width * 0.15),
                        int(height * 0.15),
                        int(width * 0.85),
                        int(height * 0.85),
                    ),
                ),
                ("BOTTOM", (0, int(height * 0.80), width, height)),
            ]
        )
    elif crops_mode != "none":
        raise ValueError(f"Unsupported crops mode: {crops_mode}")

    for i, crop in enumerate(manual_crops or [], start=1):
        clipped = _clip_box(crop, width, height)
        if clipped is not None:
            boxes.append((f"MANUAL_{i:02d}", clipped))
    return boxes


def _text_len(text: str) -> int:
    compact = re.sub(r"\s+", " ", text).strip()
    return len(compact)


def _ocr_with_psm(image: Image.Image, lang: str, oem: int, psm: int) -> tuple[str, float | None]:
    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pytesseract is not installed. Run: pip install -r requirements.txt") from exc

    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    confs: list[float] = []
    for conf_str in data.get("conf", []):
        try:
            conf = float(conf_str)
        except (TypeError, ValueError):
            continue
        if conf >= 0:
            confs.append(conf)
    avg_conf = round(sum(confs) / len(confs), 2) if confs else None
    return text, avg_conf


def _ocr_best_pass(
    image: Image.Image,
    lang: str,
    oem: int,
    psm: int,
    second_psm: int | None,
) -> tuple[str, float | None, int]:
    primary_text, primary_conf = _ocr_with_psm(image, lang=lang, oem=oem, psm=psm)
    best = (primary_text, primary_conf, psm)
    if second_psm is None or second_psm == psm:
        return best
    secondary_text, secondary_conf = _ocr_with_psm(image, lang=lang, oem=oem, psm=second_psm)
    if _text_len(secondary_text) > _text_len(primary_text):
        return secondary_text, secondary_conf, second_psm
    return best


def _ocr_worker(task: dict) -> dict:
    frame_path = Path(task["frame_path"])
    timestamp = float(task["timestamp"])
    lang = task["lang"]
    psm = int(task["psm"])
    second_psm = task["second_psm"]
    oem = int(task["oem"])
    crops_mode = task["crops_mode"]
    manual_crops = task["manual_crops"]

    with Image.open(frame_path) as img:
        image = img.convert("L")
        crops = build_crop_boxes(image.width, image.height, crops_mode=crops_mode, manual_crops=manual_crops)

        crop_records: list[dict] = []
        for crop_name, box in crops:
            crop_img = image.crop(box)
            text, avg_conf, used_psm = _ocr_best_pass(
                crop_img, lang=lang, oem=oem, psm=psm, second_psm=second_psm
            )
            crop_records.append(
                {
                    "name": crop_name,
                    "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                    "text": text,
                    "avg_conf": avg_conf,
                    "psm_used": used_psm,
                }
            )

    parts = [f"[{record['name']}]\n{record['text']}" for record in crop_records if record["text"]]
    full_text = "\n\n".join(parts).strip()

    valid_confs = [record["avg_conf"] for record in crop_records if record["avg_conf"] is not None]
    avg_conf = round(sum(valid_confs) / len(valid_confs), 2) if valid_confs else None

    return {
        "timestamp": round(timestamp, 3),
        "frame_path": str(frame_path.resolve()),
        "ocr_provider": "tesseract",
        "provider_meta": {
            "lang": lang,
            "oem": oem,
            "psm": psm,
            "second_psm": second_psm,
        },
        "ocr": {
            "full_text": full_text,
            "avg_conf": avg_conf,
            "crops": crop_records,
        },
    }


def dedupe_frames(
    frame_entries: list[dict],
    *,
    time_gap_sec: float = 2.0,
    hamming_threshold: int = 6,
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    dropped: list[dict] = []
    recent_kept: list[dict] = []

    for entry in sorted(frame_entries, key=lambda item: item["timestamp"]):
        ts = float(entry["timestamp"])
        processed_path = Path(entry["processed_path"])
        try:
            with Image.open(processed_path) as img:
                current_hash = imagehash.phash(img.convert("L"))
        except Exception as exc:
            kept.append({**entry, "phash": None, "phash_error": str(exc)})
            continue

        duplicate_of: dict | None = None
        hash_distance: int | None = None
        for prev in reversed(recent_kept):
            if ts - prev["timestamp"] > time_gap_sec:
                break
            distance = current_hash - prev["hash"]
            if distance <= hamming_threshold:
                duplicate_of = prev
                hash_distance = int(distance)
                break

        if duplicate_of is not None:
            dropped.append(
                {
                    **entry,
                    "phash": str(current_hash),
                    "dropped_reason": "near_duplicate",
                    "duplicate_of": duplicate_of["processed_path"],
                    "hash_distance": hash_distance,
                }
            )
            continue

        kept_entry = {**entry, "phash": str(current_hash)}
        kept.append(kept_entry)
        recent_kept.append({"timestamp": ts, "hash": current_hash, "processed_path": entry["processed_path"]})
        recent_kept = [prev for prev in recent_kept if ts - prev["timestamp"] <= time_gap_sec]

    return kept, dropped


def ocr_frames(frame_entries: list[dict], options: OcrOptions, workers: int = 4) -> list[dict]:
    if not frame_entries:
        return []

    max_workers = max(1, min(int(workers), os.cpu_count() or 1))
    tasks = [
        {
            "timestamp": float(entry["timestamp"]),
            "frame_path": entry["processed_path"],
            "lang": options.lang,
            "psm": options.psm,
            "second_psm": options.second_psm,
            "oem": options.oem,
            "crops_mode": options.crops_mode,
            "manual_crops": options.manual_crops,
        }
        for entry in frame_entries
    ]

    total = len(tasks)
    log(f"Running OCR on {total} frames with {max_workers} worker(s).")

    if max_workers == 1:
        out: list[dict] = []
        for i, task in enumerate(tasks, start=1):
            out.append(_ocr_worker(task))
            if i % 5 == 0 or i == total:
                log(f"OCR progress: {i}/{total}")
        return sorted(out, key=lambda item: item["timestamp"])

    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_ocr_worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            out.append(future.result())
            if i % 5 == 0 or i == total:
                log(f"OCR progress: {i}/{total}")
    return sorted(out, key=lambda item: item["timestamp"])
