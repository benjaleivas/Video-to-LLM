from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .utils import ensure_dir, format_seconds_for_filename, log


@dataclass
class PreprocessOptions:
    scale: float = 2.0
    threshold: str = "adaptive"  # none | otsu | adaptive
    image_format: str = "png"  # png | jpg
    denoise: bool = True
    sharpen: bool = True


def _threshold_array(gray_array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return gray_array
    if mode == "otsu":
        _, out = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
    raise ValueError(f"Unsupported threshold mode: {mode}")


def preprocess_image(raw_path: Path, processed_path: Path, options: PreprocessOptions) -> None:
    image = Image.open(raw_path).convert("RGB")

    if options.scale and options.scale > 0 and abs(options.scale - 1.0) > 1e-6:
        new_size = (
            max(1, int(round(image.width * options.scale))),
            max(1, int(round(image.height * options.scale))),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    if options.sharpen:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=175, threshold=3))

    gray_array = np.array(gray)
    if options.denoise:
        gray_array = cv2.medianBlur(gray_array, 3)

    out_array = _threshold_array(gray_array, options.threshold)
    out_img = Image.fromarray(out_array)

    ext = options.image_format.lower()
    if ext in {"jpg", "jpeg"}:
        out_img.convert("L").save(processed_path, format="JPEG", quality=95, optimize=True)
    elif ext == "png":
        out_img.save(processed_path, format="PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported image format: {options.image_format}")


def preprocess_frames(
    frame_entries: list[dict],
    frames_out_dir: Path,
    options: PreprocessOptions,
) -> list[dict]:
    ensure_dir(frames_out_dir)
    total = len(frame_entries)
    out_entries: list[dict] = []

    for i, entry in enumerate(sorted(frame_entries, key=lambda item: item["timestamp"]), start=1):
        ts = float(entry["timestamp"])
        ext = options.image_format.lower()
        if ext == "jpeg":
            ext = "jpg"
        filename = f"frame_{format_seconds_for_filename(ts)}.{ext}"
        processed_path = (frames_out_dir / filename).resolve()
        preprocess_image(Path(entry["raw_path"]), processed_path, options)

        if i % 10 == 0 or i == total:
            log(f"Preprocess progress: {i}/{total}")

        out_entries.append({**entry, "processed_path": str(processed_path)})
    return out_entries
