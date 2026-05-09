"""Turn a finished image generation into a Runway custom avatar.

Pipeline:

    1. Look up the generation row (must be ``status='succeeded'`` and have
       a saved image on disk plus a stored identity package).
    2. Upload the image bytes via ``client.uploads.create_ephemeral`` to
       get an HTTPS URL Runway can fetch.
    3. Call ``client.avatars.create`` with the persona's name, personality,
       voice preset and the uploaded reference image, plus the start
       script the LLM wrote.
    4. Persist the returned avatar id on the generation row, so the UI
       can link to a chat page that streams that custom avatar.

Avatar creation is a foreground operation as far as the SDK is concerned
(`avatars.create` blocks until the avatar exists), but we still run it
off the FastAPI thread so the request returns immediately and the UI
polls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runwayml import RunwayML

from . import knowledge, storage

log = logging.getLogger("scout.avatars")


def _client() -> RunwayML:
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    if not api_key:
        raise RuntimeError(
            "RUNWAYML_API_KEY (or RUNWAYML_API_SECRET) is not set"
        )
    return RunwayML(api_key=api_key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upload_reference(client: RunwayML, image_path: Path) -> str:
    """Upload an image and return a Runway-resolvable reference.

    ``uploads.create_ephemeral`` returns a ``runway://...`` URI (a signed
    JWT pointing at the file in Runway's S3 bucket). Other Runway
    endpoints — including ``avatars.create`` — accept this URI as a
    drop-in for ``reference_image`` even though the public docs say
    "HTTPS URL". The URI expires after roughly 24h, which is plenty for
    avatar creation since Runway copies the bytes server-side.
    """
    with image_path.open("rb") as fh:
        upload = client.uploads.create_ephemeral(file=fh)
    ref = getattr(upload, "uri", None) or getattr(upload, "url", None)
    if not ref:
        raise RuntimeError(f"Upload returned no usable reference: {upload!r}")
    return ref


def create_avatar_for_generation(gen_id: str) -> dict:
    """Create a Runway avatar from generation ``gen_id``.

    Returns the updated DB record. On any failure the row's
    ``avatar_status`` is set to ``failed`` and the error stored.
    """
    storage.update_generation(
        gen_id, avatar_status="creating", avatar_error=None
    )
    try:
        rec = storage.get_generation(gen_id)
        if not rec:
            raise LookupError(f"generation {gen_id} not found")
        if rec.get("status") != "succeeded":
            raise RuntimeError(
                f"generation status is '{rec.get('status')}', need 'succeeded'"
            )
        if not rec.get("image_path"):
            raise RuntimeError("generation has no image_path")
        for key in ("character_name", "personality", "voice_preset"):
            if not rec.get(key):
                raise RuntimeError(
                    f"generation is missing identity field '{key}' — "
                    "regenerate it with the new prompts service"
                )

        image_path = storage.IMAGES_DIR / rec["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"image file missing: {image_path}")

        client = _client()
        log.info("uploading reference image gen=%s path=%s", gen_id, image_path.name)
        reference_url = _upload_reference(client, image_path)
        log.info("uploaded gen=%s -> %s", gen_id, reference_url[:80])

        # Personality and start_script are passed to the avatar at *create*
        # time — Runway stores them and uses them as the default for every
        # realtime session that uses this avatar id.
        kwargs: dict = {
            "name": rec["character_name"][:64],
            "personality": rec["personality"],
            "reference_image": reference_url,
            "voice": {
                "type": "runway-live-preset",
                "preset_id": rec["voice_preset"],
            },
        }
        if rec.get("start_script"):
            kwargs["start_script"] = rec["start_script"]

        log.info(
            "creating runway avatar gen=%s name=%r voice=%s",
            gen_id,
            kwargs["name"],
            rec["voice_preset"],
        )
        created = client.avatars.create(**kwargs)
        avatar_id = getattr(created, "id", None) or getattr(created, "avatar_id", None)
        if not avatar_id:
            raise RuntimeError(f"Runway returned no avatar id: {created!r}")

        storage.update_generation(
            gen_id,
            runway_avatar_id=avatar_id,
            avatar_status="ready",
            avatar_error=None,
            avatar_created_at=_now(),
        )
        log.info("gen=%s avatar created id=%s", gen_id, avatar_id)

        # Chain knowledge ingestion immediately so a single user click
        # ("Make this an avatar") gives them an avatar that *knows* the
        # project. Failures are isolated — the avatar is still usable
        # without a knowledge base, just less informed.
        try:
            knowledge.attach_knowledge_for_generation(gen_id)
        except Exception:
            log.exception("gen=%s knowledge ingestion errored after avatar create", gen_id)

        return storage.get_generation(gen_id) or {}
    except Exception as e:
        log.exception("gen=%s avatar creation failed", gen_id)
        storage.update_generation(
            gen_id,
            avatar_status="failed",
            avatar_error=f"{type(e).__name__}: {e}"[:1000],
        )
        return storage.get_generation(gen_id) or {}
