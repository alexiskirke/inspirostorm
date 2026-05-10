"""Generate one short Runway video from two reference images + a prompt.

The user-facing pattern is: take the two avatar reference STILLS we
already produced (stored on disk under ``DATA_DIR/images``), upload them
to Runway as ephemeral references, and call ``image_to_video`` with one
image as the FIRST frame and the other as the LAST frame. Runway's
model fills the in-between motion driven by the synth-derived prompt.

Default model is ``gen3a_turbo`` (cheapest at 5 credits/sec → ~$0.50
for a 10-sec clip), upgradable to ``veo3.1_fast`` ($1/10s, no audio) or
``veo3.1`` with audio ($4/10s) via the ``model`` arg.

This module is deliberately decoupled from the brainstorm DB so it can
be exercised standalone — see ``scout/scripts/test_movie.py``. The
brainstorm-pipeline wrapper that records movie_status / movie_path on a
synthesis row lives in ``services/brainstorm.py`` (see
``trigger_movie_for_synthesis`` in a follow-up patch — not in this
file).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests
from runwayml import RunwayML

from . import storage

log = logging.getLogger("scout.movies")

# Conservative defaults. Each model accepts a different set of ratios
# and durations — Runway will 400 if you pick something out of range:
#   gen3a_turbo : ratios {16:9, 9:16, 768:1280, 1280:768}, duration {5, 10}
#   gen4_turbo  : ratios {1280:720, 720:1280, ...},        duration {5, 10}
#   gen4.5      : ratios {1280:720, 720:1280, 1080:1920, 1920:1080}, duration {4, 6, 8}
#   veo3.1_fast : ratios {16:9, 9:16},                     duration 8
#   veo3.1      : ratios {16:9, 9:16},                     duration 8
# We default to the cheapest pair (gen3a_turbo + 1280:768 + 10s).
DEFAULT_MODEL = os.environ.get("MOVIE_MODEL", "gen3a_turbo")
DEFAULT_DURATION = int(os.environ.get("MOVIE_DURATION", "10"))
DEFAULT_RATIO = os.environ.get("MOVIE_RATIO", "1280:768")
POLL_TIMEOUT_S = int(os.environ.get("MOVIE_TIMEOUT_S", "600"))


# ---------------------------------------------------------------------------


@dataclass
class MovieResult:
    """The artefact + everything we need to log, persist or render."""

    output_path: Path                 # local MP4
    runway_task_id: str
    runway_output_url: Optional[str]  # the (ephemeral) signed URL we downloaded
    prompt: str
    model: str
    duration: int
    ratio: str
    bytes_written: int = 0
    references: list[str] = field(default_factory=list)  # runway:// uris we used


# ---------------------------------------------------------------------------


def _client() -> RunwayML:
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    if not api_key:
        raise RuntimeError(
            "RUNWAYML_API_KEY (or RUNWAYML_API_SECRET) is not set"
        )
    return RunwayML(api_key=api_key)


def _upload_image(client: RunwayML, path: Path) -> str:
    """Upload an image and return Runway's resolvable reference URI.

    Same trick as in ``avatars.create_avatar_for_generation`` — the SDK
    returns a ``runway://`` URI that other Runway endpoints accept.
    """
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with path.open("rb") as fh:
        upload = client.uploads.create_ephemeral(file=fh)
    ref = getattr(upload, "uri", None) or getattr(upload, "url", None)
    if not ref:
        raise RuntimeError(f"upload returned no usable reference: {upload!r}")
    return ref


def _download_output(url: str, dest: Path) -> int:
    """Stream the Runway output URL to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = 0
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                total += len(chunk)
    return total


def generate_video(
    image_paths: Iterable[Path],
    prompt: str,
    *,
    output_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    duration: int = DEFAULT_DURATION,
    ratio: str = DEFAULT_RATIO,
    position_mode: str = "keyframes",
    audio: Optional[bool] = None,
) -> MovieResult:
    """End-to-end: upload images → start image_to_video task → poll →
    download MP4 → return MovieResult.

    ``position_mode`` controls how images are passed:

      "keyframes" (default — works on every model that has
                   ``position`` Required): each image gets a position.
        - 1 image  → that image is the FIRST frame
        - 2 images → first → FIRST frame, second → LAST frame
                     (the "morph between" pattern: avatar A on frame 1,
                     avatar B on the final frame, prompt drives the
                     in-between)

      "references" (currently only supported by ``seedance2``): position
                   is OMITTED so the model treats each image as a pure
                   reference. The prompt assigns roles ("character on
                   the left looks like image 1, character on the right
                   like image 2"). Up to 9 images allowed by Seedance.

    ``>2`` images are silently capped (keyframes mode keeps the first 2;
    references mode keeps the first 9).
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("at least one reference image is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    if position_mode not in {"keyframes", "references"}:
        raise ValueError(f"position_mode must be 'keyframes' or 'references', got {position_mode!r}")

    client = _client()
    log.info(
        "uploading %d reference image(s) for image_to_video model=%s "
        "duration=%ds ratio=%s mode=%s",
        len(paths), model, duration, ratio, position_mode,
    )
    cap = 9 if position_mode == "references" else 2
    refs: list[str] = []
    for p in paths[:cap]:
        refs.append(_upload_image(client, p))

    if position_mode == "references":
        # Omit position entirely — Runway treats each image as a pure
        # visual reference. Only seedance2 currently accepts this shape.
        prompt_image = [{"uri": uri} for uri in refs]
    elif len(refs) == 1:
        prompt_image = [{"position": "first", "uri": refs[0]}]
    else:
        prompt_image = [
            {"position": "first", "uri": refs[0]},
            {"position": "last", "uri": refs[1]},
        ]

    log.info(
        "creating runway image_to_video task model=%s audio=%s prompt=%r",
        model, audio,
        prompt[:120] + ("…" if len(prompt) > 120 else ""),
    )
    create_kwargs: dict = {
        "model": model,
        "prompt_image": prompt_image,
        "prompt_text": prompt,
        "ratio": ratio,
        "duration": duration,
    }
    # Only include `audio` when the caller explicitly opted in/out — for
    # models that don't accept the flag (e.g. gen3a_turbo) we'd 400 with
    # an unknown-field error if we always sent it.
    if audio is not None:
        create_kwargs["audio"] = audio
    task = client.image_to_video.create(**create_kwargs)              # type: ignore[arg-type]
    task_id = getattr(task, "id", None) or "?"
    log.info("runway task %s created, polling…", task_id)

    deadline = time.time() + POLL_TIMEOUT_S
    last_status = ""
    result_obj = None
    while time.time() < deadline:
        t = client.tasks.retrieve(task_id)
        status = getattr(t, "status", None)
        if status != last_status:
            log.info("runway task %s status=%s", task_id, status)
            last_status = status or ""
        if status == "SUCCEEDED":
            result_obj = t
            break
        if status in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"runway task {status}: {getattr(t, 'failure', None) or status}"
            )
        time.sleep(3.0)
    else:
        raise TimeoutError(
            f"runway task {task_id} did not finish within {POLL_TIMEOUT_S}s"
        )

    outputs = getattr(result_obj, "output", None) or []
    if not outputs:
        raise RuntimeError("runway task succeeded but returned no output URL")
    output_url = outputs[0] if isinstance(outputs[0], str) else getattr(
        outputs[0], "url", None
    )
    if not output_url:
        raise RuntimeError(f"could not extract output URL: {outputs!r}")

    if output_path is None:
        output_path = storage.MOVIES_DIR / f"{task_id}.mp4"
    log.info("downloading mp4 from runway → %s", output_path)
    n_bytes = _download_output(output_url, output_path)
    log.info("downloaded %d bytes", n_bytes)

    return MovieResult(
        output_path=output_path,
        runway_task_id=task_id,
        runway_output_url=output_url,
        prompt=prompt,
        model=model,
        duration=duration,
        ratio=ratio,
        bytes_written=n_bytes,
        references=refs,
    )
