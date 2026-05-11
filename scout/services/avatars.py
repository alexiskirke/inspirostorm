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

from . import knowledge, prompts, storage

log = logging.getLogger("scout.avatars")


def _voices_taken_by_other_custom_avatars(this_gen_id: str) -> set[str]:
    """Return the set of voice_preset ids already in use by other
    generations that have been promoted to a Runway custom avatar
    (``runway_avatar_id`` not null). Used to detect cross-batch voice
    collisions when this generation is about to be promoted itself."""
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT voice_preset FROM generations
             WHERE runway_avatar_id IS NOT NULL
               AND runway_avatar_id <> ''
               AND id <> ?
               AND voice_preset IS NOT NULL
               AND voice_preset <> ''
            """,
            (this_gen_id,),
        ).fetchall()
    return {r["voice_preset"] for r in rows}


def _reassign_voice_if_collision(gen_id: str, current_voice: str) -> str:
    """Layer-2 safety net: if ``current_voice`` is already used by another
    custom avatar, swap to an unused voice from the same gender bucket
    (falling back to neutral, then to any unused id). Persists the swap
    on the DB row.

    Returns the (possibly new) voice id. If no collision OR no replacement
    is available, returns ``current_voice`` unchanged.
    """
    taken = _voices_taken_by_other_custom_avatars(gen_id)
    if current_voice not in taken:
        return current_voice
    gender = prompts.VOICE_BY_ID.get(current_voice, {}).get("gender", "neutral")
    candidates = prompts.voices_for_gender(gender, exclude=taken)
    if not candidates:
        # gender-specific + neutral both exhausted — fall back to any unused
        candidates = [v["id"] for v in prompts.VOICE_OPTIONS if v["id"] not in taken]
    if not candidates:
        log.warning(
            "gen=%s voice collision on %r but no replacement available; "
            "keeping the duplicate", gen_id, current_voice,
        )
        return current_voice
    new_voice = candidates[0]
    log.info(
        "gen=%s cross-batch voice collision: %r is taken; reassigning to %r "
        "(same gender bucket: %s)", gen_id, current_voice, new_voice, gender,
    )
    storage.update_generation(gen_id, voice_preset=new_voice)
    return new_voice


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

        # Cross-batch voice-collision guard: if another generation has
        # already become a Runway custom avatar with this same voice
        # preset, swap to an unused same-gender voice and persist the
        # swap. (Layer 1 batch-coord in /api/generate covers same-batch
        # collisions; this catches the cross-batch case the user
        # actually hit — two avatars made separately ending up with
        # the same voice.)
        effective_voice = _reassign_voice_if_collision(gen_id, rec["voice_preset"])

        # Personality and start_script are passed to the avatar at *create*
        # time — Runway stores them and uses them as the default for every
        # realtime session that uses this avatar id.
        kwargs: dict = {
            "name": rec["character_name"][:64],
            "personality": rec["personality"],
            "reference_image": reference_url,
            "voice": {
                "type": "runway-live-preset",
                "preset_id": effective_voice,
            },
        }
        if rec.get("start_script"):
            kwargs["start_script"] = rec["start_script"]

        log.info(
            "creating runway avatar gen=%s name=%r voice=%s",
            gen_id,
            kwargs["name"],
            effective_voice,
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


def delete_runway_artifacts(gen_id: str) -> dict:
    """Best-effort teardown of the Runway-side artifacts owned by this
    generation: the custom character and every document attached to it.

    Never raises — failures are logged and reported in the return dict
    so the caller can decide whether to surface them. Local DB / disk
    cleanup is the caller's responsibility (storage.delete_generation).
    """
    rec = storage.get_generation(gen_id)
    if not rec:
        return {"avatar_deleted": False, "documents_deleted": 0, "errors": ["generation not found"]}

    errors: list[str] = []
    avatar_id = rec.get("runway_avatar_id")
    doc_ids_csv = rec.get("runway_document_ids") or ""
    doc_ids = [d for d in doc_ids_csv.split(",") if d.strip()]

    if not avatar_id and not doc_ids:
        return {"avatar_deleted": False, "documents_deleted": 0, "errors": []}

    try:
        client = _client()
    except Exception as e:
        # No key → can't talk to Runway. Surface the error but allow the
        # local delete to proceed (the user explicitly asked to delete).
        return {
            "avatar_deleted": False,
            "documents_deleted": 0,
            "errors": [f"{type(e).__name__}: {e}"],
        }

    avatar_deleted = False
    if avatar_id:
        try:
            client.avatars.delete(avatar_id)
            avatar_deleted = True
            log.info("gen=%s runway avatar %s deleted", gen_id, avatar_id)
        except Exception as e:
            msg = f"avatar {avatar_id}: {type(e).__name__}: {e}"
            log.warning("gen=%s runway avatar delete failed: %s", gen_id, msg)
            errors.append(msg)

    documents_deleted = 0
    for did in doc_ids:
        try:
            client.documents.delete(did)
            documents_deleted += 1
        except Exception as e:
            msg = f"document {did}: {type(e).__name__}: {e}"
            log.warning("gen=%s runway document delete failed: %s", gen_id, msg)
            errors.append(msg)
    if doc_ids:
        log.info(
            "gen=%s runway documents deleted: %d/%d",
            gen_id, documents_deleted, len(doc_ids),
        )

    return {
        "avatar_deleted": avatar_deleted,
        "documents_deleted": documents_deleted,
        "documents_total": len(doc_ids),
        "errors": errors,
    }
