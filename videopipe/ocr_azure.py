from __future__ import annotations

from io import BytesIO
import os
from typing import Any

from PIL import Image


def _collect_read_lines(payload: dict[str, Any]) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    confs: list[float] = []

    read_result = payload.get("readResult") or {}
    blocks = read_result.get("blocks") or []
    for block in blocks:
        lines = block.get("lines") or []
        for line in lines:
            text = str(line.get("text") or "").strip()
            if text:
                texts.append(text)
            words = line.get("words") or []
            for word in words:
                conf = word.get("confidence")
                if isinstance(conf, (int, float)):
                    value = float(conf)
                    confs.append(value * 100.0 if value <= 1.0 else value)
    return texts, confs


def ocr_crop_azure_read(
    crop_img: Image.Image,
    *,
    timeout_seconds: float = 30.0,
    endpoint: str | None = None,
    api_key: str | None = None,
) -> tuple[str, float | None, str | None, dict[str, Any]]:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - import guard
        return "", None, f"requests dependency missing: {exc}", {}

    endpoint = (endpoint or os.getenv("AZURE_VISION_ENDPOINT") or "").strip()
    api_key = (api_key or os.getenv("AZURE_VISION_KEY") or "").strip()
    if not endpoint or not api_key:
        return "", None, "AZURE_VISION_ENDPOINT or AZURE_VISION_KEY not configured", {}

    url = (
        endpoint.rstrip("/")
        + "/computervision/imageanalysis:analyze"
        + "?api-version=2023-10-01&features=read"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/octet-stream",
    }

    with BytesIO() as buf:
        crop_img.save(buf, format="PNG")
        payload = buf.getvalue()

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=timeout_seconds)
    except Exception as exc:  # pragma: no cover - network/runtime
        return "", None, str(exc), {}

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except Exception:
            error_payload = {"text": response.text}
        return "", None, f"HTTP {response.status_code}: {error_payload}", {"status_code": response.status_code}

    try:
        data = response.json()
    except Exception as exc:
        return "", None, f"Invalid JSON response: {exc}", {}

    lines, confs = _collect_read_lines(data)
    text = "\n".join(lines).strip()
    avg_conf = round(sum(confs) / len(confs), 2) if confs else None
    return text, avg_conf, None, {"status_code": response.status_code}
