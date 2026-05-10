"""Smoke test for ``scout.services.movies.generate_video``.

Two modes — pick whichever is more convenient:

  Mode A (point at two image files directly):
    venv/bin/python -m scout.scripts.test_movie \
        --image scout/data/images/AAA.png \
        --image scout/data/images/BBB.png \
        --prompt "Slow cinematic morph from one to the other..."

  Mode B (use Scout generation ids; we look up image_path on disk):
    venv/bin/python -m scout.scripts.test_movie \
        --gen-id 8b52880283dd4a9894cd42518184b388 \
        --gen-id <other-gen-id> \
        --prompt "..."

Mode C (cheapest one-image test — verifies the pipe with minimum credit
burn before risking a two-frame run):
    venv/bin/python -m scout.scripts.test_movie \
        --gen-id 8b52880283dd4a9894cd42518184b388 \
        --duration 5 \
        --prompt "..."

Optional flags: --model gen3a_turbo|veo3.1_fast|veo3.1, --duration 5|10,
--ratio 1280:720|768:1280, --output some/path.mp4.

This script *deliberately does not touch the brainstorm DB*. It just
proves the Runway round-trip — uploads → image_to_video task → poll →
download MP4.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "scout" / ".env")

from scout.services import movies, storage  # noqa: E402


def _resolve_gen_image(gen_id: str) -> Path:
    """Look up the image_path for a Scout generation row and return its
    on-disk Path."""
    db = storage.DB_PATH
    if not db.exists():
        raise FileNotFoundError(f"scout DB not found: {db}")
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT character_name, image_path FROM generations WHERE id = ?",
        (gen_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise LookupError(f"generation {gen_id!r} not in DB")
    name, image_path = row
    if not image_path:
        raise LookupError(f"generation {gen_id!r} has no image_path")
    full = storage.IMAGES_DIR / image_path
    if not full.exists():
        raise FileNotFoundError(f"image not on disk: {full}")
    print(f"  gen {gen_id[:8]} ({name}) → {full}")
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", action="append", default=[],
        help="Path to a reference image (max 2). Repeatable.",
    )
    parser.add_argument(
        "--gen-id", action="append", default=[],
        help="Scout generation id (looks up image_path on disk). Repeatable; max 2.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for the video")
    parser.add_argument("--model", default=movies.DEFAULT_MODEL,
                        help=f"Runway model (default: {movies.DEFAULT_MODEL})")
    parser.add_argument("--duration", type=int, default=movies.DEFAULT_DURATION,
                        help=f"Seconds (default: {movies.DEFAULT_DURATION})")
    parser.add_argument("--ratio", default=movies.DEFAULT_RATIO,
                        help=f"Aspect ratio (default: {movies.DEFAULT_RATIO})")
    parser.add_argument("--output", help="Output mp4 path (default: data/movies/<task_id>.mp4)")
    parser.add_argument(
        "--mode", choices=["keyframes", "references"], default="keyframes",
        help="Image positioning. keyframes = first/last frames (works on most "
             "models). references = omit position (Seedance2 only — both images "
             "act as character references in the same scene).",
    )
    parser.add_argument(
        "--audio", action="store_true",
        help="Enable audio (speech / music). Only Veo 3 / Veo 3.1 family. "
             "Not allowed on gen3a_turbo, gen4_turbo, gen4.5, seedance2.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    image_paths: list[Path] = []
    print("--- resolving images ---")
    for p in args.image:
        full = Path(p).expanduser().resolve()
        if not full.exists():
            print(f"ERROR: image not found: {full}", file=sys.stderr)
            return 2
        print(f"  file → {full}")
        image_paths.append(full)
    for gid in args.gen_id:
        try:
            image_paths.append(_resolve_gen_image(gid))
        except (LookupError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    if not image_paths:
        print("ERROR: pass at least one --image or --gen-id", file=sys.stderr)
        return 2
    if len(image_paths) > 2:
        print(f"WARNING: only the first 2 images will be used "
              f"(got {len(image_paths)})", file=sys.stderr)

    out: Path | None = Path(args.output).expanduser().resolve() if args.output else None

    print()
    print("--- calling Runway image_to_video ---")
    print(f"  model    : {args.model}")
    print(f"  duration : {args.duration}s")
    print(f"  ratio    : {args.ratio}")
    print(f"  mode     : {args.mode}")
    print(f"  prompt   : {args.prompt[:140]}{'…' if len(args.prompt) > 140 else ''}")
    print()

    try:
        result = movies.generate_video(
            image_paths,
            args.prompt,
            output_path=out,
            model=args.model,
            duration=args.duration,
            ratio=args.ratio,
            position_mode=args.mode,
            audio=True if args.audio else None,
        )
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print("--- success ---")
    print(f"  output_path     : {result.output_path}")
    print(f"  bytes_written   : {result.bytes_written:,}")
    print(f"  runway_task_id  : {result.runway_task_id}")
    print(f"  references_used : {len(result.references)}")
    print()
    print(f"To view: open '{result.output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
