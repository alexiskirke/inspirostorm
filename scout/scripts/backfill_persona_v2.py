"""One-shot back-fill: regenerate persona v2 fields for every Scout
custom avatar, push the rewritten personality to Runway, and store the
new ``domain_body`` / ``domain_summary`` / ``weirdness`` columns.

Idempotent: re-running just refreshes the personality. Skips rows that
have no ``runway_avatar_id`` (i.e. images that were never turned into a
Runway avatar).

Usage:
    venv/bin/python -m scout.scripts.backfill_persona_v2 [--gen-id ID]

With no args it iterates all eligible rows; with ``--gen-id`` it only
touches that single row (handy for one-off testing).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "scout" / ".env")

from runwayml import RunwayML  # noqa: E402

from scout.services import prompts, storage  # noqa: E402

log = logging.getLogger("scout.backfill")


def _runway() -> RunwayML:
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    if not api_key:
        raise RuntimeError("RUNWAYML_API_KEY not set")
    return RunwayML(api_key=api_key)


def _eligible_rows(only_gen_id: Optional[str] = None) -> list[dict]:
    rows = storage.list_generations(limit=1000)
    out: list[dict] = []
    for r in rows:
        if only_gen_id and r["id"] != only_gen_id:
            continue
        if not r.get("runway_avatar_id"):
            continue
        if r.get("status") != "succeeded":
            continue
        out.append(r)
    return out


def _backfill_one(rec: dict, client: RunwayML) -> bool:
    gen_id = rec["id"]
    log.info("backfilling gen=%s name=%s", gen_id, rec.get("character_name"))

    source = rec.get("source_meta")
    if not isinstance(source, dict):
        log.warning("gen=%s missing source_meta; skipping", gen_id)
        return False

    readme = rec.get("readme_excerpt") or ""
    weirdness = float(rec.get("weirdness") or 0.33)

    try:
        identity = prompts.generate_identity(
            source, readme=readme, weirdness=weirdness
        )
    except Exception as e:
        log.exception("gen=%s identity regeneration failed: %s", gen_id, e)
        return False

    # Persist the new building blocks back to Scout DB. We DO update
    # start_script now too — the older stored openers were written in
    # "service-mode" (Sage's "Tell me what meditation you're making…")
    # which collapses brainstorm sessions into 1-on-1 coaching the
    # moment a custom avatar's stored startScript wins. The new
    # persona-generation prompt explicitly bans that wording.
    # Voice and image_path are left untouched so the avatar's look and
    # sound stay stable across back-fills.
    storage.update_generation(
        gen_id,
        domain_body=identity["domain_body"],
        domain_summary=identity["domain_summary"],
        weirdness=identity["weirdness"],
        personality=identity["personality"],
        start_script=identity["start_script"],
    )

    # Push the new personality + start_script to Runway. Other fields
    # (voice/image/etc.) unchanged.
    try:
        client.avatars.update(
            rec["runway_avatar_id"],
            personality=identity["personality"],
            start_script=identity["start_script"],
        )
    except Exception as e:
        log.exception("gen=%s avatars.update failed: %s", gen_id, e)
        return False

    log.info(
        "gen=%s OK (personality %d chars, domain_summary %d chars)\n"
        "  new start_script: %s",
        gen_id,
        len(identity["personality"]),
        len(identity["domain_summary"]),
        identity["start_script"],
    )
    return True


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-id", help="Only back-fill this generation id")
    args = parser.parse_args()

    client = _runway()
    rows = _eligible_rows(only_gen_id=args.gen_id)
    if not rows:
        log.warning("no eligible avatars found")
        return 1

    log.info("back-filling %d avatar(s)", len(rows))
    ok = sum(1 for r in rows if _backfill_one(r, client))
    log.info("done: %d/%d succeeded", ok, len(rows))
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
