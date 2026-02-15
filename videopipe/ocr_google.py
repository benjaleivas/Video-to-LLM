from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .ocr import OcrOptions, build_crop_boxes
from .utils import log


def _lang_hints(lang: str) -> list[str] | None:
    value = (lang or "").strip().lower()
    if not value:
        return None
    mapping = {
        "eng": "en",
        "en": "en",
        "spa": "es",
        "es": "es",
        "fra": "fr",
        "fr": "fr",
        "deu": "de",
        "de": "de",
    }
    hint = mapping.get(value, value)
    return [hint]


def _collect_confidences(annotation: Any) -> list[float]:
    confs: list[float] = []
    if not annotation:
        return confs

    pages = getattr(annotation, "pages", None) or []
    for page in pages:
        page_conf = getattr(page, "confidence", None)
        if isinstance(page_conf, (int, float)) and page_conf >= 0:
            confs.append(float(page_conf) * 100.0 if page_conf <= 1 else float(page_conf))

        blocks = getattr(page, "blocks", None) or []
        for block in blocks:
            block_conf = getattr(block, "confidence", None)
            if isinstance(block_conf, (int, float)) and block_conf >= 0:
                confs.append(float(block_conf) * 100.0 if block_conf <= 1 else float(block_conf))

            paragraphs = getattr(block, "paragraphs", None) or []
            for para in paragraphs:
                para_conf = getattr(para, "confidence", None)
                if isinstance(para_conf, (int, float)) and para_conf >= 0:
                    confs.append(float(para_conf) * 100.0 if para_conf <= 1 else float(para_conf))

                words = getattr(para, "words", None) or []
                for word in words:
                    word_conf = getattr(word, "confidence", None)
                    if isinstance(word_conf, (int, float)) and word_conf >= 0:
                        confs.append(float(word_conf) * 100.0 if word_conf <= 1 else float(word_conf))

                    symbols = getattr(word, "symbols", None) or []
                    for symbol in symbols:
                        sym_conf = getattr(symbol, "confidence", None)
                        if isinstance(sym_conf, (int, float)) and sym_conf >= 0:
                            confs.append(float(sym_conf) * 100.0 if sym_conf <= 1 else float(sym_conf))
    return confs


def _ocr_crop_google(
    client: Any,
    crop_img: Image.Image,
    feature: str,
    timeout_seconds: float,
    language_hints: list[str] | None,
) -> tuple[str, float | None, str | None]:
    from google.cloud import vision

    with BytesIO() as buf:
        crop_img.save(buf, format="PNG")
        content = buf.getvalue()

    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=language_hints) if language_hints else None

    try:
        if feature == "document_text_detection":
            response = client.document_text_detection(
                image=image,
                image_context=image_context,
                timeout=timeout_seconds,
            )
            if response.error.message:
                return "", None, response.error.message
            annotation = response.full_text_annotation
            text = (annotation.text or "").strip() if annotation else ""
            confs = _collect_confidences(annotation)
            avg_conf = round(sum(confs) / len(confs), 2) if confs else None
            return text, avg_conf, None

        if feature == "text_detection":
            response = client.text_detection(
                image=image,
                image_context=image_context,
                timeout=timeout_seconds,
            )
            if response.error.message:
                return "", None, response.error.message
            text = ""
            if response.text_annotations:
                text = (response.text_annotations[0].description or "").strip()

            annotation = getattr(response, "full_text_annotation", None)
            confs = _collect_confidences(annotation)
            avg_conf = round(sum(confs) / len(confs), 2) if confs else None
            return text, avg_conf, None

        return "", None, f"Unsupported google OCR feature: {feature}"
    except Exception as exc:  # pragma: no cover - network/runtime
        return "", None, str(exc)


def _ocr_worker(task: dict) -> dict:
    from google.cloud import vision

    frame_path = Path(task["frame_path"])
    timestamp = float(task["timestamp"])
    feature = task["feature"]
    timeout_seconds = float(task["timeout_seconds"])
    language_hints = task["language_hints"]
    options: OcrOptions = task["options"]

    client = vision.ImageAnnotatorClient()

    with Image.open(frame_path) as img:
        image = img.convert("L")
        crops = build_crop_boxes(image.width, image.height, crops_mode=options.crops_mode, manual_crops=options.manual_crops)

        crop_records: list[dict] = []
        errors: list[dict] = []
        for crop_name, box in crops:
            crop_img = image.crop(box)
            text, avg_conf, error = _ocr_crop_google(
                client=client,
                crop_img=crop_img,
                feature=feature,
                timeout_seconds=timeout_seconds,
                language_hints=language_hints,
            )
            if error:
                errors.append({"crop": crop_name, "error": error})

            crop_records.append(
                {
                    "name": crop_name,
                    "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                    "text": text,
                    "avg_conf": avg_conf,
                    "psm_used": None,
                    "error": error,
                }
            )

    parts = [f"[{record['name']}]\n{record['text']}" for record in crop_records if record["text"]]
    full_text = "\n\n".join(parts).strip()
    valid_confs = [record["avg_conf"] for record in crop_records if record.get("avg_conf") is not None]
    avg_conf = round(sum(valid_confs) / len(valid_confs), 2) if valid_confs else None

    return {
        "timestamp": round(timestamp, 3),
        "frame_path": str(frame_path.resolve()),
        "ocr_provider": "google",
        "provider_meta": {
            "feature": feature,
            "language_hints": language_hints,
            "timeout_seconds": timeout_seconds,
            "errors": errors,
        },
        "ocr": {
            "full_text": full_text,
            "avg_conf": avg_conf,
            "crops": crop_records,
        },
    }


def ocr_frames_google(
    frame_entries: list[dict],
    options: OcrOptions,
    *,
    feature: str = "document_text_detection",
    timeout_seconds: float = 30.0,
    max_concurrency: int = 4,
) -> list[dict]:
    if not frame_entries:
        return []

    if feature not in {"document_text_detection", "text_detection"}:
        raise ValueError("feature must be one of: document_text_detection, text_detection")

    workers = max(1, int(max_concurrency))
    language_hints = _lang_hints(options.lang)

    tasks = [
        {
            "timestamp": float(entry["timestamp"]),
            "frame_path": entry["processed_path"],
            "feature": feature,
            "timeout_seconds": timeout_seconds,
            "language_hints": language_hints,
            "options": options,
        }
        for entry in frame_entries
    ]

    total = len(tasks)
    log(f"Running Google OCR on {total} frames with {workers} worker(s).")

    if workers == 1:
        out: list[dict] = []
        for i, task in enumerate(tasks, start=1):
            out.append(_ocr_worker(task))
            if i % 5 == 0 or i == total:
                log(f"Google OCR progress: {i}/{total}")
        return sorted(out, key=lambda item: item["timestamp"])

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_ocr_worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            out.append(future.result())
            if i % 5 == 0 or i == total:
                log(f"Google OCR progress: {i}/{total}")

    return sorted(out, key=lambda item: item["timestamp"])
