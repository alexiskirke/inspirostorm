"""Generate avatar images via Runway and persist them to disk."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests
from runwayml import RunwayML

from . import storage

log = logging.getLogger("scout.images")

DEFAULT_MODEL = os.environ.get("RUNWAY_IMAGE_MODEL", "gen4_image")
DEFAULT_RATIO = os.environ.get("RUNWAY_IMAGE_RATIO", "1024:1024")
GENERATION_TIMEOUT_S = int(os.environ.get("RUNWAY_IMAGE_TIMEOUT", "300"))


def _client() -> RunwayML:
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    if not api_key:
        raise RuntimeError(
            "RUNWAYML_API_KEY (or RUNWAYML_API_SECRET) is not set"
        )
    return RunwayML(api_key=api_key)


def _download(url: str) -> tuple[bytes, str]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/png")
    ext = "png"
    if "jpeg" in content_type or "jpg" in content_type:
        ext = "jpg"
    elif "webp" in content_type:
        ext = "webp"
    return resp.content, ext


def generate_and_store(
    gen_id: str,
    *,
    prompt: str,
    model: Optional[str] = None,
    ratio: Optional[str] = None,
) -> dict:
    """Run a Runway image generation end-to-end and persist the result.

    Returns the updated DB record. On any error the record is marked 'failed'
    and the exception message is stored.
    """
    chosen_model = model or DEFAULT_MODEL
    chosen_ratio = ratio or DEFAULT_RATIO
    try:
        client = _client()
        log.info(
            "creating runway image task gen=%s model=%s ratio=%s",
            gen_id,
            chosen_model,
            chosen_ratio,
        )
        task = client.text_to_image.create(
            model=chosen_model,
            prompt_text=prompt[:1000],
            ratio=chosen_ratio,
        )
        runway_task_id = getattr(task, "id", None) or "unknown"
        storage.mark_running(gen_id, runway_task_id)
        log.info("runway task %s created (gen=%s), polling…", runway_task_id, gen_id)

        deadline = time.time() + GENERATION_TIMEOUT_S
        result = None
        while time.time() < deadline:
            t = client.tasks.retrieve(runway_task_id)
            status = getattr(t, "status", None)
            if status == "SUCCEEDED":
                result = t
                break
            if status in {"FAILED", "CANCELLED"}:
                msg = getattr(t, "failure", None) or status or "unknown"
                raise RuntimeError(f"Runway task {status}: {msg}")
            time.sleep(2.0)
        else:
            raise TimeoutError(
                f"Runway task {runway_task_id} did not finish within "
                f"{GENERATION_TIMEOUT_S}s"
            )

        outputs = getattr(result, "output", None) or []
        if not outputs:
            raise RuntimeError("Runway task succeeded but returned no output URL")
        image_url = outputs[0] if isinstance(outputs[0], str) else getattr(
            outputs[0], "url", None
        )
        if not image_url:
            raise RuntimeError(f"Could not extract image URL from output: {outputs!r}")

        content, ext = _download(image_url)
        path = storage.save_image_bytes(gen_id, content, ext=ext)
        storage.mark_succeeded(gen_id, path)
        log.info("gen=%s saved %s (%d bytes)", gen_id, path, len(content))
        return storage.get_generation(gen_id) or {}
    except Exception as e:
        log.exception("gen=%s failed", gen_id)
        storage.mark_failed(gen_id, f"{type(e).__name__}: {e}")
        return storage.get_generation(gen_id) or {}
