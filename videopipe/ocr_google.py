from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .ocr import OcrOptions, build_crop_boxes
from .ocr_azure import ocr_crop_azure_read
from .preprocess import build_google_crop_variants
from .quality import score_ocr_text
from .utils import log_verbose, make_progress


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
            confs.append(
                float(page_conf) * 100.0 if page_conf <= 1 else float(page_conf)
            )

        blocks = getattr(page, "blocks", None) or []
        for block in blocks:
            block_conf = getattr(block, "confidence", None)
            if isinstance(block_conf, (int, float)) and block_conf >= 0:
                confs.append(
                    float(block_conf) * 100.0 if block_conf <= 1 else float(block_conf)
                )

            paragraphs = getattr(block, "paragraphs", None) or []
            for para in paragraphs:
                para_conf = getattr(para, "confidence", None)
                if isinstance(para_conf, (int, float)) and para_conf >= 0:
                    confs.append(
                        float(para_conf) * 100.0 if para_conf <= 1 else float(para_conf)
                    )

                words = getattr(para, "words", None) or []
                for word in words:
                    word_conf = getattr(word, "confidence", None)
                    if isinstance(word_conf, (int, float)) and word_conf >= 0:
                        confs.append(
                            float(word_conf) * 100.0
                            if word_conf <= 1
                            else float(word_conf)
                        )

                    symbols = getattr(word, "symbols", None) or []
                    for symbol in symbols:
                        sym_conf = getattr(symbol, "confidence", None)
                        if isinstance(sym_conf, (int, float)) and sym_conf >= 0:
                            confs.append(
                                float(sym_conf) * 100.0
                                if sym_conf <= 1
                                else float(sym_conf)
                            )
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
    image_context = (
        vision.ImageContext(language_hints=language_hints) if language_hints else None
    )

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


def _pick_best_candidate(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda item: (
            float(item.get("quality_score") or 0.0),
            len(str(item.get("text") or "")),
            float(item.get("avg_conf") or 0.0),
        ),
    )


def _build_crop_record(
    *,
    crop_name: str,
    box: tuple[int, int, int, int],
    candidates: list[dict],
    selected: dict,
    fallback_used: bool,
) -> dict:
    return {
        "name": crop_name,
        "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "text": selected.get("text") or "",
        "avg_conf": selected.get("avg_conf"),
        "psm_used": None,
        "error": selected.get("error"),
        "quality_score": selected.get("quality_score"),
        "quality_flags": selected.get("quality_flags") or [],
        "quality_metrics": selected.get("quality_metrics") or {},
        "fallback_used": bool(fallback_used),
        "provider_candidates": candidates,
    }


def _ocr_worker(task: dict) -> dict:
    from google.cloud import vision

    frame_path = Path(task["frame_path"])
    raw_path = Path(task.get("raw_path") or frame_path)
    timestamp = float(task["timestamp"])
    feature = task["feature"]
    timeout_seconds = float(task["timeout_seconds"])
    language_hints = task["language_hints"]
    options: OcrOptions = task["options"]
    quality_threshold = float(task.get("quality_threshold", 0.55))
    fallback_provider = str(task.get("fallback_provider") or "none")

    client = vision.ImageAnnotatorClient()

    base_img_path = raw_path if raw_path.exists() else frame_path
    with Image.open(base_img_path) as img:
        image = img.convert("RGB")
        crops = build_crop_boxes(
            image.width,
            image.height,
            crops_mode=options.crops_mode,
            manual_crops=options.manual_crops,
        )

        crop_records: list[dict] = []
        errors: list[dict] = []
        frame_candidates: list[dict] = []
        fallback_count = 0

        for crop_name, box in crops:
            crop_img = image.crop(box)
            variants = build_google_crop_variants(crop_img)

            candidates: list[dict] = []

            # Pass 1: Google variant A
            text_a, conf_a, err_a = _ocr_crop_google(
                client=client,
                crop_img=variants["google_a"],
                feature=feature,
                timeout_seconds=timeout_seconds,
                language_hints=language_hints,
            )
            score_a = score_ocr_text(text_a, conf_a)
            cand_a = {
                "provider": "google",
                "variant": "google_a",
                "text": text_a,
                "avg_conf": conf_a,
                "error": err_a,
                "quality_score": score_a["quality_score"],
                "quality_flags": score_a["quality_flags"],
                "quality_metrics": score_a["metrics"],
            }
            candidates.append(cand_a)

            best = cand_a

            # Pass 2: Google variant B if low quality or provider error.
            if err_a or float(cand_a["quality_score"]) < quality_threshold:
                text_b, conf_b, err_b = _ocr_crop_google(
                    client=client,
                    crop_img=variants["google_b"],
                    feature=feature,
                    timeout_seconds=timeout_seconds,
                    language_hints=language_hints,
                )
                score_b = score_ocr_text(text_b, conf_b)
                cand_b = {
                    "provider": "google",
                    "variant": "google_b",
                    "text": text_b,
                    "avg_conf": conf_b,
                    "error": err_b,
                    "quality_score": score_b["quality_score"],
                    "quality_flags": score_b["quality_flags"],
                    "quality_metrics": score_b["metrics"],
                }
                candidates.append(cand_b)
                best = _pick_best_candidate(candidates)

            used_fallback = False
            # Fallback provider for low quality crops.
            if fallback_provider == "azure" and (
                float(best.get("quality_score") or 0.0) < quality_threshold
                or best.get("error")
            ):
                text_z, conf_z, err_z, meta_z = ocr_crop_azure_read(
                    variants["google_b"],
                    timeout_seconds=timeout_seconds,
                )
                score_z = score_ocr_text(text_z, conf_z)
                cand_z = {
                    "provider": "azure",
                    "variant": "azure_read",
                    "text": text_z,
                    "avg_conf": conf_z,
                    "error": err_z,
                    "quality_score": score_z["quality_score"],
                    "quality_flags": score_z["quality_flags"],
                    "quality_metrics": score_z["metrics"],
                    "meta": meta_z,
                }
                candidates.append(cand_z)
                best = _pick_best_candidate(candidates)
                used_fallback = best.get("provider") == "azure"
                if used_fallback:
                    fallback_count += 1

            if best.get("error"):
                errors.append({"crop": crop_name, "error": best["error"]})

            frame_candidates.extend(candidates)
            crop_records.append(
                _build_crop_record(
                    crop_name=crop_name,
                    box=box,
                    candidates=candidates,
                    selected=best,
                    fallback_used=used_fallback,
                )
            )

    parts = [
        f"[{record['name']}]\n{record['text']}"
        for record in crop_records
        if record["text"]
    ]
    full_text = "\n\n".join(parts).strip()
    valid_confs = [
        record["avg_conf"]
        for record in crop_records
        if record.get("avg_conf") is not None
    ]
    avg_conf = round(sum(valid_confs) / len(valid_confs), 2) if valid_confs else None
    frame_quality = score_ocr_text(full_text, avg_conf)

    out = {
        "timestamp": round(timestamp, 3),
        "frame_path": str(frame_path.resolve()),
        "ocr_provider": "google",
        "provider_meta": {
            "feature": feature,
            "language_hints": language_hints,
            "timeout_seconds": timeout_seconds,
            "errors": errors,
            "per_crop_errors": {item["crop"]: item["error"] for item in errors},
            "fallback_provider": fallback_provider,
        },
        "ocr": {
            "full_text": full_text,
            "avg_conf": avg_conf,
            "crops": crop_records,
        },
        "quality_score": frame_quality["quality_score"],
        "quality_flags": frame_quality["quality_flags"],
        "quality_metrics": frame_quality["metrics"],
        "fallback_used": fallback_count > 0,
        "provider_candidates": frame_candidates,
    }
    return out


def ocr_frames_google(
    frame_entries: list[dict],
    options: OcrOptions,
    *,
    feature: str = "document_text_detection",
    timeout_seconds: float = 30.0,
    max_concurrency: int = 4,
    quality_threshold: float = 0.55,
    fallback_provider: str = "none",
) -> list[dict]:
    if not frame_entries:
        return []

    if feature not in {"document_text_detection", "text_detection"}:
        raise ValueError(
            "feature must be one of: document_text_detection, text_detection"
        )
    if fallback_provider not in {"none", "azure"}:
        raise ValueError("fallback_provider must be one of: none, azure")

    workers = max(1, int(max_concurrency))
    language_hints = _lang_hints(options.lang)

    tasks = [
        {
            "timestamp": float(entry["timestamp"]),
            "frame_path": entry["processed_path"],
            "raw_path": entry.get("raw_path"),
            "feature": feature,
            "timeout_seconds": timeout_seconds,
            "language_hints": language_hints,
            "options": options,
            "quality_threshold": quality_threshold,
            "fallback_provider": fallback_provider,
        }
        for entry in frame_entries
    ]

    total = len(tasks)
    log_verbose(f"Running Google OCR on {total} frames with {workers} worker(s).")

    fallback_label = f" + {fallback_provider}" if fallback_provider != "none" else ""
    out: list[dict] = []

    with make_progress(f"OCR (Google{fallback_label})") as progress:
        bar = progress.add_task("ocr", total=total)

        if workers == 1:
            for task in tasks:
                out.append(_ocr_worker(task))
                progress.advance(bar)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_ocr_worker, task) for task in tasks]
                for future in as_completed(futures):
                    out.append(future.result())
                    progress.advance(bar)

    return sorted(out, key=lambda item: item["timestamp"])
