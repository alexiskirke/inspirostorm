"""Thread-level idea synthesis.

The summariser (gpt-5.1) is a record-keeper — it writes per-session
memory the avatars will read next time. The synthesiser (gpt-5.5) does
the creative leap: given EVERYTHING in a brainstorm thread (rolling
state + every session's transcript + every session's idea log + both
personas), produce:

  - a deep ``synthesis`` markdown doc:
      * the core idea / problem the brainstorm has converged on
      * the strongest sub-ideas with rationale
      * surprising or unconventional angles raised
      * open questions / counterpoints / risks
      * suggested next moves
  - a one-sentence ``movie_pitch`` that downstream phase 7 will render
    into a 10-sec video starring both avatars
  - a structured ``ideas`` array (the "best of" — the synth's own
    edited shortlist, distinct from the summariser's per-session lists)

A row is appended to ``brainstorm_synthesis`` each time so you can
re-run as the thread evolves and compare.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from . import brainstorm, storage

log = logging.getLogger("scout.synthesis")

DEFAULT_MODEL = os.environ.get("SYNTHESIS_MODEL", "gpt-5.5")
FALLBACK_MODELS = ["gpt-5.2", "gpt-5", "gpt-4.1", "gpt-4o"]

# We're feeding it the WHOLE thread, so allow more context than the
# summariser. Cap to keep cost predictable.
MAX_TRANSCRIPT_CHARS_PER_SESSION = 8000

SYSTEM_PROMPT = """You are a senior research-and-product strategist who
synthesises raw brainstorm transcripts into deeper, surprising ideas.

You will receive:
  - the personas of two AI characters who held a series of brainstorm
    sessions (their domains and the angles they typically bring)
  - every prior session's transcripts AND structured idea logs
  - the current "rolling memory" the characters share

Your job is NOT to recap. It is to do the creative leap the avatars
themselves haven't fully made yet:
  1. Identify the most interesting throughline across all sessions —
     the question, problem, or possibility space the brainstorm has
     converged on (often implicit; you have to name it).
  2. Pick the 4-7 strongest specific sub-ideas. Rephrase each in your
     own words; cite which character (or the human) raised the seed.
     For each, add ONE concrete next move someone could take this week.
  3. Surface the genuinely unconventional moves the avatars made (or
     should have made). Be willing to amplify weird ideas if they have
     hidden merit; reject safe ones charitably.
  4. List the open questions / counterpoints / risks you'd push the
     pair to chase next session.
  5. Compose ONE single-sentence movie pitch (max 30 words) that could
     be staged as a 10-second video starring the two characters,
     embodying the brainstorm's distilled idea. It should be visual,
     concrete, and intriguing — not a mission statement.

Return ONLY a single JSON object:

  synthesis    string   1500-3000 chars of markdown. Use sections
                        (## headers) for: Throughline, Strongest ideas,
                        Unconventional angles, Open questions, Suggested
                        next moves. Write in clear, opinionated prose.
  ideas        array    4-7 items. Each: { "headline": "...",
                                            "detail": "...",
                                            "next_move": "...",
                                            "raised_by": "<character or 'human' or 'joint'>" }
  movie_pitch  string   ONE sentence, max 30 words. The single best
                        visual pitch.

Hard rules:
- Output a single JSON object. No prose before or after. No markdown
  fences. No comments inside the JSON.
- Use double quotes for strings. Escape newlines inside strings as \\n.
- Be concrete, not vague. Name files/concepts when they were named in
  the transcripts. Avoid corporate-speak.
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_session(sess: dict) -> str:
    """Render a session as 'topic + a's transcript + b's transcript + idea log'."""
    parts: list[str] = []
    parts.append(f"### Session {sess['id'][:8]} (started {sess.get('started_at')})")
    if sess.get("topic"):
        parts.append(f"Topic: {sess['topic']}")
    for slot in ("a", "b"):
        raw = sess.get(f"transcript_{slot}") or sess.get(f"transcript_{slot}_json") or []
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except Exception: raw = []
        lines: list[str] = []
        for e in raw:
            if not isinstance(e, dict) or e.get("final") is False:
                continue
            text = (e.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"  {e.get('participantIdentity', '?')}: {text}")
        block = "\n".join(lines)
        if len(block) > MAX_TRANSCRIPT_CHARS_PER_SESSION:
            block = block[:MAX_TRANSCRIPT_CHARS_PER_SESSION] + "\n  …[truncated]"
        parts.append(f"Transcript (slot {slot}):\n{block or '  (empty)'}")
    return "\n\n".join(parts)


def _build_user_prompt(thread: dict, sessions: list[dict], state: Optional[dict],
                       avatar_a: dict, avatar_b: dict) -> str:
    parts: list[str] = []
    parts.append(
        f"CHARACTER A: {avatar_a.get('character_name', '?')}"
        f"\n  Domain: {(avatar_a.get('domain_summary') or '').strip()}"
    )
    parts.append(
        f"CHARACTER B: {avatar_b.get('character_name', '?')}"
        f"\n  Domain: {(avatar_b.get('domain_summary') or '').strip()}"
    )
    if thread.get("topic_seed"):
        parts.append(f"THREAD TOPIC SEED: {thread['topic_seed']}")
    if state and state.get("rolling_summary"):
        parts.append("CURRENT ROLLING MEMORY:\n" + state["rolling_summary"])
        if state.get("ideas"):
            parts.append("ROLLING IDEA LOG:\n" + json.dumps(state["ideas"], indent=2)[:3000])
    parts.append(f"NUMBER OF SESSIONS IN THIS THREAD: {len(sessions)}")
    # Include the most recent N sessions oldest→newest (so the model sees
    # progression). Reversed because list_sessions returns newest first.
    for sess in reversed(sessions[-6:]):
        parts.append(_format_session(sess))
    parts.append("Now produce the JSON synthesis object.")
    return "\n\n".join(parts)


def _parse_output(raw: str) -> dict:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in synthesis output: {raw[:200]}")
    obj = json.loads(text[start : end + 1])
    for key in ("synthesis", "movie_pitch"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f"synthesis field {key!r} missing/empty")
    if not isinstance(obj.get("ideas"), list):
        raise ValueError("ideas must be an array")
    return obj


def synthesise_thread(thread_id: str) -> dict:
    """Run the gpt-5.5 synthesiser and append a brainstorm_synthesis row."""
    thread = brainstorm.get_thread(thread_id)
    if not thread:
        raise LookupError(f"thread {thread_id} not found")
    avatar_a = storage.get_generation(thread["avatar_a_gen_id"])
    avatar_b = storage.get_generation(thread["avatar_b_gen_id"])
    if not avatar_a or not avatar_b:
        raise LookupError("missing avatar(s) for thread")
    sessions = brainstorm.list_sessions(thread_id)
    if not sessions:
        raise RuntimeError("no sessions yet — nothing to synthesise")
    state = brainstorm.get_thread_state(thread_id)

    user_msg = _build_user_prompt(thread, sessions, state, avatar_a, avatar_b)

    candidates = [DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != DEFAULT_MODEL]
    last_err: Optional[Exception] = None
    client = _client()
    for m in candidates:
        try:
            kwargs: dict = {
                "model": m,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
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
                last_err = RuntimeError(f"model {m} returned empty content")
                continue
            parsed = _parse_output(text)

            syn_id = uuid.uuid4().hex
            with storage._LOCK, storage._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO brainstorm_synthesis (
                        id, thread_id, scope, source_session_id,
                        text_md, movie_pitch, ideas_json,
                        model_used, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        syn_id, thread_id, "thread", None,
                        parsed["synthesis"], parsed["movie_pitch"],
                        json.dumps(parsed["ideas"]),
                        m, _now(),
                    ),
                )
            log.info(
                "synthesis ok thread=%s model=%s syn_id=%s pitch=%r",
                thread_id, m, syn_id, parsed["movie_pitch"][:60],
            )
            return {
                "id": syn_id,
                "thread_id": thread_id,
                "scope": "thread",
                "text_md": parsed["synthesis"],
                "movie_pitch": parsed["movie_pitch"],
                "ideas": parsed["ideas"],
                "model_used": m,
                "created_at": _now(),
            }
        except Exception as e:
            log.warning("synthesis model %s failed: %s", m, e)
            last_err = e
            continue
    raise RuntimeError(f"synthesis failed across {candidates}: {last_err}")


def list_synthesis(thread_id: str) -> list[dict]:
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM brainstorm_synthesis WHERE thread_id = ? "
            "ORDER BY created_at DESC",
            (thread_id,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        if d.get("ideas_json"):
            try:
                d["ideas"] = json.loads(d["ideas_json"])
            except Exception:
                d["ideas"] = []
        out.append(d)
    return out
