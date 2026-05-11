"""End-to-end movie generation from a brainstorm synthesis row.

Produces a 24-second three-shot mini-movie (8s + 8s + 8s) with each
character delivering a spoken line. Dialogue only — no music
underscore (Veo's own ambient room tone is kept).

Pipeline (per ``synthesis_id``):

  1. Look up the synthesis + thread + both avatar generations.
  2. Ask gpt-5.1 to produce a structured "shoot plan":
        - 3 clips: (Smith solo) → (Scholar solo) → (both together)
        - per-clip composite_prompt + speech_prompt
  3. For each clip:
        a. Generate the composite still via gen4_image multi-ref
           (composites/movies services already exist — we just call them).
           Solo clips use 1 reference; duo clip uses both.
        b. Generate the speech-only Veo3.1_fast (audio=True) clip from
           that composite.
  4. ffmpeg-concat the 3 Veo clips into the final 24-sec video.
  5. Persist final MP4 + status fields onto ``brainstorm_synthesis``.

Design notes:
  - Per-clip composites cost 5 credits each (gen4_image @ 1280:720), Veo
    clips cost 125 each (15 cr/sec × 8 sec + audio surcharge).
    Total: ~390 credits per finished movie. The user is on a 50k tier.
  - All intermediate files (composites, per-clip Veo outputs) are kept
    on disk under ``DATA_DIR/composites`` and ``DATA_DIR/movies`` for
    diagnostics.
  - The "NO MUSIC AT ALL" instruction baked into each speech_prompt is
    aimed at Veo (which otherwise tries to invent its own score) — it
    is unrelated to the now-removed pipeline-level music step. Keep it.
  - On any error, ``brainstorm_synthesis.movie_status='failed'`` and
    ``movie_error`` is populated — the API never raises out of this
    module (callers run it inside a background executor).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI

from . import composites, movies, storage

log = logging.getLogger("scout.movie_pipeline")

# -----------------------------------------------------------------------------
# Config (env-overridable)
# -----------------------------------------------------------------------------

PLANNER_MODEL = os.environ.get("MOVIE_PLANNER_MODEL", "gpt-5.1")
PLANNER_FALLBACKS = ["gpt-5", "gpt-4.1", "gpt-4o"]

VIDEO_MODEL = os.environ.get("MOVIE_VIDEO_MODEL", "veo3.1_fast")
VIDEO_RATIO = os.environ.get("MOVIE_VIDEO_RATIO", "1280:720")
VIDEO_DURATION_S = int(os.environ.get("MOVIE_CLIP_DURATION_S", "8"))
COMPOSITE_RATIO = os.environ.get("MOVIE_COMPOSITE_RATIO", "1280:720")


# -----------------------------------------------------------------------------
# Step 2: gpt-5.1 shoot planner
# -----------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a director planning a 24-second three-shot
mini-movie that distills a brainstorm between two characters into one
visual idea.

You will be given:
  - the brainstorm THREAD context (the optional guide prompt, the
    thread's topic seed)
  - the SYNTHESIS produced by gpt-5.4 (a deeper idea + movie_pitch +
    structured ideas)
  - both CHARACTERS' personas (name, voice, what they're an expert in,
    a one-paragraph domain summary)

Plan three 8-second shots in this fixed order:
  1. CHARACTER_A solo (full body, action). They deliver one short line
     that introduces the brainstorm's core idea from THEIR angle.
  2. CHARACTER_B solo (full body, action). They reply with one short
     line that adds THEIR angle.
  3. BOTH characters together (dynamic two-shot). They jointly deliver
     the punchline / movie pitch — either alternating one phrase each,
     or one of them speaks while the other reacts visibly.

Return ONLY one JSON object with this exact shape:

  clips         array  EXACTLY 3 items, in the order above. Each item:
                       {
                         title:           short label,
                         speaker:         "a" | "b" | "both",
                         image_strategy:  "solo_a" | "solo_b" | "duo",
                         composite_prompt: prompt for gen4_image
                           multi-reference (use @character_a /
                           @character_b tags as appropriate). Describe
                           full-body, dynamic framing, the
                           setting/environment, lighting, art style.
                           For solo clips, include only the relevant
                           character's tag. For duo, include both.
                           MUST be ≤ 800 characters.
                         speech_prompt: 3-5 sentence prompt for Veo
                           3.1 fast. Include the EXACT line(s) of
                           dialogue in quotes. CRITICAL: end with
                           "NO MUSIC AT ALL in the audio. Just clear
                           spoken dialogue plus quiet ambient room
                           tone. Cinematic camera, no on-screen text."
                           MUST be ≤ 800 characters TOTAL — that
                           includes the trailing NO MUSIC sentence.
                           If you can't fit a full description, cut
                           prose before cutting dialogue. The duo
                           clip's joint dialogue counts toward the
                           same 800 char budget.
                       }

Hard rules:
- Output ONE JSON object, no prose around it, no markdown fences.
- Tags MUST match exactly @{character_a_tag} and @{character_b_tag} —
  use the placeholders given to you in the user message.
- Each speech_prompt MUST contain the dialogue lines in double quotes
  so Veo speaks them verbatim.
- LENGTH (Runway hard-rejects ``promptText`` > 1000 chars): every
  composite_prompt and speech_prompt MUST be ≤ 800 characters. Stay
  well clear — clip 3's combined two-character dialogue is the most
  likely offender. Keep prose tight; the dialogue is the priority.
- Keep all text safe-for-work, no real-person names, no copyrighted IP.
- If the brainstorm guide / topic seed asks for a specific deliverable
  (e.g. "propose a new web app" or "co-write a screenplay logline"),
  ground the punchline in clip 3 in that deliverable.
"""

# Runway's text_to_image / image_to_video endpoints reject promptText
# over 1000 chars (HTTP 400 from validation). Leave headroom for tail
# additions and tag substitution.
RUNWAY_PROMPT_MAX_CHARS = 990


def _openai() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_plan(raw: str) -> dict:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError(f"no JSON in planner output: {raw[:200]}")
    obj = json.loads(text[s : e + 1])
    if not isinstance(obj.get("clips"), list) or len(obj["clips"]) != 3:
        raise ValueError("planner must return exactly 3 clips")
    for i, c in enumerate(obj["clips"]):
        for key in ("title", "speaker", "image_strategy", "composite_prompt", "speech_prompt"):
            if not isinstance(c.get(key), str) or not c[key].strip():
                raise ValueError(f"clip {i} missing field {key!r}")
    return obj


_NO_MUSIC_RE = re.compile(r"\bNO MUSIC AT ALL\b.*$", re.IGNORECASE | re.DOTALL)
_FIRST_QUOTE_RE = re.compile(r'(["“„«][^"”’»]{4,}["”’»])', re.DOTALL)


def _trim_at_word(text: str, limit: int) -> str:
    """Hard-truncate at the last whitespace boundary before ``limit``."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Try not to break mid-word; back off to the last whitespace.
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:—-") + "…"


def _clamp_speech_prompt(prompt: str, *, limit: int = RUNWAY_PROMPT_MAX_CHARS) -> str:
    """Keep a Veo speech_prompt under ``limit`` chars while preserving
    the parts that *must* survive:
      - the dialogue (first quoted block, the line Veo will actually speak)
      - the trailing NO MUSIC sentence

    We do this by trimming the descriptive prose between the start and
    the dialogue (or between the dialogue and the NO MUSIC tail), since
    that's the lowest-signal segment.
    """
    if len(prompt) <= limit:
        return prompt

    tail_match = _NO_MUSIC_RE.search(prompt)
    tail = tail_match.group(0) if tail_match else ""
    body = prompt[: tail_match.start()] if tail_match else prompt

    quote_match = _FIRST_QUOTE_RE.search(body)
    dialogue = quote_match.group(1) if quote_match else ""

    # If we already exceed the budget without the prose, accept that
    # — at minimum we need dialogue + tail. Truncate dialogue last.
    must_keep = (dialogue + " " + tail).strip()
    if len(must_keep) >= limit:
        # Pathological: dialogue + tail alone are too long. Hard-trim.
        return _trim_at_word(must_keep, limit)

    # Budget for the prose around the dialogue.
    overhead_budget = limit - len(must_keep) - 4  # 4 chars: ellipsis + spacing
    prefix = body[: quote_match.start()] if quote_match else body
    suffix = body[quote_match.end():] if quote_match else ""

    # Allocate budget proportionally to original lengths.
    total_prose = len(prefix) + len(suffix)
    if total_prose <= overhead_budget:
        return prompt  # shouldn't happen — len check at top already
    if total_prose:
        prefix_budget = max(40, int(overhead_budget * (len(prefix) / total_prose)))
        suffix_budget = max(0, overhead_budget - prefix_budget)
    else:
        prefix_budget, suffix_budget = overhead_budget, 0

    new_prefix = _trim_at_word(prefix, prefix_budget) if prefix else ""
    new_suffix = _trim_at_word(suffix, suffix_budget) if suffix else ""
    rebuilt = " ".join(p for p in [new_prefix, dialogue, new_suffix, tail] if p).strip()
    # Final defensive hard-trim in case our budget math undershot.
    if len(rebuilt) > limit:
        rebuilt = _trim_at_word(rebuilt, limit)
    return rebuilt


def _clamp_for_runway(
    prompt: str,
    *,
    kind: str,
    limit: int = RUNWAY_PROMPT_MAX_CHARS,
    label: str = "",
) -> str:
    """Ensure ``prompt`` fits Runway's promptText limit.

    ``kind`` is one of ``"composite"`` or ``"speech"`` — speech prompts
    use structure-preserving truncation, composites just tail-trim.
    Logs a warning when truncation fires so we can spot planner drift.
    """
    if len(prompt) <= limit:
        return prompt
    if kind == "speech":
        out = _clamp_speech_prompt(prompt, limit=limit)
    else:
        out = _trim_at_word(prompt, limit)
    log.warning(
        "%s prompt over Runway limit (%d > %d); truncated to %d chars%s",
        kind, len(prompt), limit, len(out),
        f" [{label}]" if label else "",
    )
    return out


def plan_shoot(
    *,
    thread: dict,
    synthesis: dict,
    avatar_a: dict,
    avatar_b: dict,
    tag_a: str,
    tag_b: str,
) -> dict:
    """Ask gpt-5.1 for the 3-clip + music plan."""
    user = "\n\n".join(filter(None, [
        f"THREAD GUIDE / TOPIC: {thread.get('topic_seed') or '(none — let the synthesis lead)'}",
        f"SYNTHESIS (gpt-5.4 output):\n{(synthesis.get('text_md') or synthesis.get('text') or '').strip()}",
        f"MOVIE PITCH (one-line): {(synthesis.get('movie_pitch') or '').strip()}",
        f"IDEAS:\n{json.dumps(synthesis.get('ideas') or [], indent=2)[:4000]}",
        f"CHARACTER A — tag @{tag_a}:\n  name: {avatar_a.get('character_name')}\n  voice: {avatar_a.get('voice_preset')}\n  domain: {(avatar_a.get('domain_summary') or '').strip()}",
        f"CHARACTER B — tag @{tag_b}:\n  name: {avatar_b.get('character_name')}\n  voice: {avatar_b.get('voice_preset')}\n  domain: {(avatar_b.get('domain_summary') or '').strip()}",
        "Now produce the JSON shoot plan.",
    ]))
    client = _openai()
    last_err: Optional[Exception] = None
    for m in [PLANNER_MODEL] + [x for x in PLANNER_FALLBACKS if x != PLANNER_MODEL]:
        try:
            kwargs: dict = {
                "model": m,
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": user},
                ],
            }
            try:
                resp = client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
            except Exception:
                resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = RuntimeError(f"planner {m} returned empty content")
                continue
            plan = _parse_plan(text)
            plan["model_used"] = m
            return plan
        except Exception as e:
            log.warning("planner %s failed: %s", m, e)
            last_err = e
    raise RuntimeError(f"shoot planner failed across all models: {last_err}")


# -----------------------------------------------------------------------------
# ffmpeg helpers
# -----------------------------------------------------------------------------


def concat_videos(input_paths: list[Path], output_path: Path) -> None:
    """ffmpeg concat (uses the demuxer route — re-encodes audio so the
    AAC streams across our Veo clips line up cleanly, copies video)."""
    if not input_paths:
        raise ValueError("no input videos to concat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for p in input_paths:
            tmp.write(f"file '{p.resolve()}'\n")
        listfile = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
        log.info("ffmpeg concat → %s (%d clips)", output_path, len(input_paths))
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("ffmpeg concat failed (rc=%d):\n%s", proc.returncode, proc.stderr[-2000:])
            raise RuntimeError(f"ffmpeg concat failed (rc={proc.returncode})")
    finally:
        listfile.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_synth_movie_status(synth_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [synth_id]
    with storage._LOCK, storage._connect() as conn:
        conn.execute(f"UPDATE brainstorm_synthesis SET {cols} WHERE id = ?", vals)


def _load_synthesis(synth_id: str) -> dict:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM brainstorm_synthesis WHERE id = ?", (synth_id,)
        ).fetchone()
    if not row:
        raise LookupError(f"synthesis {synth_id} not found")
    d = dict(row)
    if d.get("ideas_json"):
        try:
            d["ideas"] = json.loads(d["ideas_json"])
        except Exception:
            d["ideas"] = []
    return d


def _load_thread(thread_id: str) -> dict:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM brainstorm_threads WHERE id = ?", (thread_id,)
        ).fetchone()
    if not row:
        raise LookupError(f"thread {thread_id} not found")
    return dict(row)


# -----------------------------------------------------------------------------
# The orchestrator
# -----------------------------------------------------------------------------


def make_movie_for_synthesis(synth_id: str) -> dict:
    """Run the full 3-clip pipeline for a brainstorm_synthesis row.

    Idempotent in the sense that it always re-runs all generations and
    overwrites the prior movie file. (We cap by the synthesis_id, not
    per-attempt — if you want a fresh attempt, run synthesise again to
    get a new synth row, or call this twice and take the second.)
    """
    log.info("== make_movie_for_synthesis %s ==", synth_id)
    _set_synth_movie_status(
        synth_id,
        movie_status="building",
        movie_error=None,
        movie_created_at=_now(),
    )
    try:
        synth = _load_synthesis(synth_id)
        thread = _load_thread(synth["thread_id"])
        avatar_a = storage.get_generation(thread["avatar_a_gen_id"])
        avatar_b = storage.get_generation(thread["avatar_b_gen_id"])
        if not (avatar_a and avatar_b):
            raise RuntimeError("missing avatar(s) for thread")
        for av, slot in [(avatar_a, "a"), (avatar_b, "b")]:
            if not av.get("image_path"):
                raise RuntimeError(f"avatar {slot} has no image_path on disk")

        tag_a = "character_a"
        tag_b = "character_b"

        # ---- Step 2: planner ----
        log.info("[%s] planning shoot via %s", synth_id, PLANNER_MODEL)
        plan = plan_shoot(
            thread=thread, synthesis=synth,
            avatar_a=avatar_a, avatar_b=avatar_b,
            tag_a=tag_a, tag_b=tag_b,
        )
        # Persist the plan early so a debug peek is possible mid-build.
        _set_synth_movie_status(
            synth_id,
            movie_prompt=json.dumps(plan)[:6000],
            movie_model=VIDEO_MODEL,
        )

        movies_dir = storage.MOVIES_DIR
        movies_dir.mkdir(parents=True, exist_ok=True)
        clip_paths: list[Path] = []
        composite_paths: list[Path] = []
        runway_task_ids: list[str] = []

        # Reusable mapping from image_strategy → list of (path, tag) refs.
        avatar_a_path = storage.IMAGES_DIR / avatar_a["image_path"]
        avatar_b_path = storage.IMAGES_DIR / avatar_b["image_path"]

        for i, clip in enumerate(plan["clips"]):
            log.info("[%s] clip %d/%d (%s) — composite", synth_id, i + 1, 3, clip["title"])
            strategy = clip["image_strategy"]
            if strategy == "solo_a":
                refs = [{"path": avatar_a_path, "tag": tag_a}]
            elif strategy == "solo_b":
                refs = [{"path": avatar_b_path, "tag": tag_b}]
            else:
                refs = [
                    {"path": avatar_a_path, "tag": tag_a},
                    {"path": avatar_b_path, "tag": tag_b},
                ]
            composite_prompt = _clamp_for_runway(
                clip["composite_prompt"],
                kind="composite",
                label=f"{synth_id} clip{i+1}",
            )
            comp = composites.make_composite(
                refs, composite_prompt,
                ratio=COMPOSITE_RATIO,
            )
            composite_paths.append(comp.output_path)

            log.info(
                "[%s] clip %d/%d (%s) — Veo %ds (audio, speech only)",
                synth_id, i + 1, 3, clip["title"], VIDEO_DURATION_S,
            )
            speech_prompt = _clamp_for_runway(
                clip["speech_prompt"],
                kind="speech",
                label=f"{synth_id} clip{i+1}",
            )
            clip_result = movies.generate_video(
                [comp.output_path], speech_prompt,
                model=VIDEO_MODEL,
                duration=VIDEO_DURATION_S,
                ratio=VIDEO_RATIO,
                position_mode="keyframes",
                audio=True,
            )
            clip_paths.append(clip_result.output_path)
            runway_task_ids.append(clip_result.runway_task_id)

        # ---- Step 4: concat clips into the final movie ----
        final_path = movies_dir / f"{synth_id}.mp4"
        log.info("[%s] concatenating %d clips → %s", synth_id, len(clip_paths), final_path)
        concat_videos(clip_paths, final_path)

        # Clean up per-clip intermediates (keep composites for diagnostics).
        for p in clip_paths:
            p.unlink(missing_ok=True)

        relname = final_path.name
        _set_synth_movie_status(
            synth_id,
            movie_status="ready",
            movie_path=relname,
            movie_runway_task_id=",".join(runway_task_ids),
            movie_error=None,
        )
        log.info("[%s] DONE — final %s", synth_id, final_path)
        return {
            "synthesis_id": synth_id,
            "movie_status": "ready",
            "movie_path": relname,
            "plan": plan,
        }
    except Exception as e:
        log.exception("make_movie_for_synthesis %s failed", synth_id)
        _set_synth_movie_status(
            synth_id,
            movie_status="failed",
            movie_error=f"{type(e).__name__}: {e}"[:1000],
        )
        return {
            "synthesis_id": synth_id,
            "movie_status": "failed",
            "movie_error": f"{type(e).__name__}: {e}",
        }


def make_movie_for_synthesis_safe(synth_id: str) -> Optional[dict]:
    try:
        return make_movie_for_synthesis(synth_id)
    except Exception:
        log.exception("safe wrapper caught: %s", synth_id)
        return None
