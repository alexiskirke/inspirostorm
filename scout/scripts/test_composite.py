"""Smoke test for ``scout.services.composites.make_composite``.

Generates a SINGLE composite still that places multiple character
references in one scene via Runway gen4_image multi-reference. This is
the upstream of the synth-movie pipeline (composite still → Veo 3.1
animation), so we verify it on its own first.

Mode A — point at images directly:
  venv/bin/python -m scout.scripts.test_composite \
      --ref scout/data/images/AAA.png:smith \
      --ref scout/data/images/BBB.png:scholar \
      --prompt "Wide cinematic shot of @smith and @scholar in a workshop..."

Mode B — use Scout generation ids (we read image_path off the DB):
  venv/bin/python -m scout.scripts.test_composite \
      --gen-ref 8b52880283dd4a9894cd42518184b388:smith \
      --gen-ref dd2ed0c19fca44b884e39f04481e7e54:scholar \
      --prompt "..."

Optional: --model gen4_image|gen4_image_turbo, --ratio 1280:720|1024:1024|...
          --output some/path.png

Cost (per the official pricing page):
  gen4_image       1280:720  → 5 credits  (~$0.05)
  gen4_image       1080p     → 8 credits  (~$0.08)
  gen4_image_turbo any       → 2 credits  (~$0.02)
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

from scout.services import composites, storage  # noqa: E402


def _parse_ref(s: str) -> tuple[str, str]:
    """Parse 'PATH_OR_GENID:tag' into (str, tag)."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--ref / --gen-ref needs PATH:tag (got {s!r})"
        )
    target, tag = s.rsplit(":", 1)
    if not target or not tag:
        raise argparse.ArgumentTypeError(
            f"--ref / --gen-ref needs non-empty path and tag (got {s!r})"
        )
    return target, tag


def _resolve_gen_image(gen_id: str) -> Path:
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
    print(f"  gen {gen_id[:8]} ({name or 'unnamed'}) → {full}")
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", action="append", default=[], type=_parse_ref,
        help="PATH:tag (repeatable). Image file path + tag.",
    )
    parser.add_argument(
        "--gen-ref", action="append", default=[], type=_parse_ref,
        help="GEN_ID:tag (repeatable). Looks up the image_path on disk.",
    )
    parser.add_argument("--prompt", required=True,
                        help="Text prompt; address each ref by @tag.")
    parser.add_argument("--model", default=composites.DEFAULT_MODEL,
                        help=f"gen4_image (default) or gen4_image_turbo")
    parser.add_argument("--ratio", default=composites.DEFAULT_RATIO,
                        help=f"Aspect ratio (default: {composites.DEFAULT_RATIO})")
    parser.add_argument("--output", help="Output png path (default: data/composites/<task_id>.png)")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    print("--- resolving references ---")
    refs: list[dict] = []
    for path_str, tag in args.ref:
        full = Path(path_str).expanduser().resolve()
        if not full.exists():
            print(f"ERROR: image not found: {full}", file=sys.stderr)
            return 2
        print(f"  file → {full} (@{tag})")
        refs.append({"path": full, "tag": tag})
    for gid, tag in args.gen_ref:
        try:
            full = _resolve_gen_image(gid)
        except (LookupError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        refs.append({"path": full, "tag": tag})
    if not refs:
        print("ERROR: pass at least one --ref or --gen-ref", file=sys.stderr)
        return 2

    out: Path | None = Path(args.output).expanduser().resolve() if args.output else None

    print()
    print("--- calling Runway gen4_image multi-reference ---")
    print(f"  model  : {args.model}")
    print(f"  ratio  : {args.ratio}")
    print(f"  refs   : {len(refs)} ({', '.join('@' + r['tag'] for r in refs)})")
    print(f"  prompt : {args.prompt[:160]}{'…' if len(args.prompt) > 160 else ''}")
    print()

    try:
        result = composites.make_composite(
            refs, args.prompt,
            output_path=out, model=args.model, ratio=args.ratio,
        )
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print("--- success ---")
    print(f"  output_path     : {result.output_path}")
    print(f"  bytes_written   : {result.bytes_written:,}")
    print(f"  runway_task_id  : {result.runway_task_id}")
    print()
    print(f"To view: open '{result.output_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
