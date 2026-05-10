"""Generate ONE multi-character composite still via Runway gen4_image
multi-reference.

The pattern: upload N reference images, give each a short tag like
``smith`` / ``scholar`` / ``lab``, and pass a prompt that addresses
them via ``@tag``. Runway composes them into a single new still.

This is the foundation for the synth movie's "both characters in one
scene" shots — that composite still then feeds into ``image_to_video``
(usually Veo 3.1 with audio) to become an animated, speaking shot.

Lives in its own module (separate from ``movies``) because the still-
gen step is reusable on its own — e.g. for thumbnails of pairs in the
brainstorm UI.
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

log = logging.getLogger("scout.composites")

DEFAULT_MODEL = os.environ.get("COMPOSITE_MODEL", "gen4_image")
DEFAULT_RATIO = os.environ.get("COMPOSITE_RATIO", "1280:720")
POLL_TIMEOUT_S = int(os.environ.get("COMPOSITE_TIMEOUT_S", "180"))

# These are derived from the gen4_image SDK signature; we keep them
# as a sanity allowlist so callers can't easily 400.
GEN4_IMAGE_RATIOS = {
    "1024:1024", "1080:1080", "1168:880", "1360:768", "1440:1080",
    "1080:1440", "1808:768", "1920:1080", "1080:1920", "2112:912",
    "1280:720", "720:1280", "720:720", "960:720", "720:960", "1680:720",
}


@dataclass
class CompositeResult:
    output_path: Path
    runway_task_id: str
    runway_output_url: Optional[str]
    prompt: str
    model: str
    ratio: str
    bytes_written: int = 0
    references: list[dict] = field(default_factory=list)


def _client() -> RunwayML:
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    if not api_key:
        raise RuntimeError(
            "RUNWAYML_API_KEY (or RUNWAYML_API_SECRET) is not set"
        )
    return RunwayML(api_key=api_key)


def _upload(client: RunwayML, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with path.open("rb") as fh:
        u = client.uploads.create_ephemeral(file=fh)
    ref = getattr(u, "uri", None) or getattr(u, "url", None)
    if not ref:
        raise RuntimeError(f"upload returned no usable reference: {u!r}")
    return ref


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = 0
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    return total


def make_composite(
    references: Iterable[dict],   # [{"path": Path|str, "tag": "smith"}, ...]
    prompt: str,
    *,
    output_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    ratio: str = DEFAULT_RATIO,
) -> CompositeResult:
    """Return a single composite still that places ALL referenced
    characters in one scene described by ``prompt``.

    The prompt should address each reference by its ``tag`` using ``@tag``
    syntax — e.g. with tags ``smith`` and ``scholar``, a prompt like
    "wide cinematic shot of @smith forging at a glowing workbench while
    @scholar sketches on a hovering chalkboard" gets you both characters.
    """
    refs = [dict(r) for r in references]
    if not refs:
        raise ValueError("at least one reference is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    if ratio not in GEN4_IMAGE_RATIOS:
        raise ValueError(
            f"ratio {ratio!r} not allowed for {model}; pick from {sorted(GEN4_IMAGE_RATIOS)}"
        )

    client = _client()

    # Upload each ref and assemble the SDK payload.
    payload_refs: list[dict] = []
    for r in refs:
        path = Path(r["path"])
        tag = (r.get("tag") or "").strip() or None
        log.info("uploading composite ref tag=%s path=%s", tag, path.name)
        uri = _upload(client, path)
        item: dict = {"uri": uri}
        if tag:
            item["tag"] = tag
        payload_refs.append(item)

    log.info(
        "creating gen4_image composite model=%s ratio=%s refs=%d prompt=%r",
        model, ratio, len(payload_refs),
        prompt[:120] + ("…" if len(prompt) > 120 else ""),
    )
    task = client.text_to_image.create(
        model=model,                         # type: ignore[arg-type]
        prompt_text=prompt,
        ratio=ratio,                         # type: ignore[arg-type]
        reference_images=payload_refs,       # type: ignore[arg-type]
    )
    task_id = getattr(task, "id", None) or "?"
    log.info("runway task %s created, polling…", task_id)

    deadline = time.time() + POLL_TIMEOUT_S
    last_status = ""
    result = None
    while time.time() < deadline:
        t = client.tasks.retrieve(task_id)
        status = getattr(t, "status", None)
        if status != last_status:
            log.info("task %s status=%s", task_id, status)
            last_status = status or ""
        if status == "SUCCEEDED":
            result = t
            break
        if status in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"task {status}: {getattr(t, 'failure', None) or status}"
            )
        time.sleep(2.0)
    else:
        raise TimeoutError(
            f"task {task_id} did not finish within {POLL_TIMEOUT_S}s"
        )

    outputs = getattr(result, "output", None) or []
    if not outputs:
        raise RuntimeError("task succeeded but returned no output URL")
    output_url = outputs[0] if isinstance(outputs[0], str) else getattr(
        outputs[0], "url", None
    )
    if not output_url:
        raise RuntimeError(f"could not extract output URL: {outputs!r}")

    # Default output: scout/data/composites/<task_id>.png
    if output_path is None:
        composites_dir = storage.DATA_DIR / "composites"
        composites_dir.mkdir(parents=True, exist_ok=True)
        output_path = composites_dir / f"{task_id}.png"
    log.info("downloading composite still → %s", output_path)
    n_bytes = _download(output_url, output_path)
    log.info("downloaded %d bytes", n_bytes)

    return CompositeResult(
        output_path=output_path,
        runway_task_id=task_id,
        runway_output_url=output_url,
        prompt=prompt,
        model=model,
        ratio=ratio,
        bytes_written=n_bytes,
        references=payload_refs,
    )
