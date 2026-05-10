"""Post-session summariser.

After every brainstorm session ends, we run a single OpenAI call
(default model: gpt-5.1, configurable via SUMMARISER_MODEL) that
condenses the latest transcript + the prior rolling memory into:

  - ``rolling_summary``  ~2k-char markdown the avatars will read at the
                         start of their next session (injected as the
                         ONGOING block by ``prompts.compose_personality``)
  - ``ideas``            structured list of concrete ideas raised (used
                         later by the gpt-5.5 thread-level synthesiser)

We deliberately use a CHEAP model here (gpt-5.1 full, not 5.5) — this
runs after every session and the work is mostly compression, not
creative leap. The expensive model is reserved for end-of-thread
synthesis.

Failure semantics:
  - We never raise out of summarise_session_safe — caller (the
    brainstorm end-session hook) keeps moving even if OpenAI is down.
  - Failures are logged + the session row is left without a
    rolling_summary, which means the next session simply starts with
    the older brainstorm_state (or no memory at all) — no data loss.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import OpenAI

from . import brainstorm, prompts, storage

log = logging.getLogger("scout.summariser")

DEFAULT_MODEL = os.environ.get("SUMMARISER_MODEL", "gpt-5.1")
FALLBACK_MODELS = ["gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]

# We cap the transcript we feed the summariser to keep cost bounded and
# avoid hitting context limits on cheaper fallback models. Realistically
# a single 5-min Runway session produces at most ~6k chars of clean
# transcript, so this is generous.
MAX_TRANSCRIPT_CHARS = 24000


SYSTEM_PROMPT = """You are a brainstorm record-keeper for a small studio.
Two AI characters had a conversation. You are given:
  - their personas (short descriptions of who each one is)
  - the most recent SESSION transcript (turn by turn)
  - optionally, the rolling MEMORY summary from previous sessions on
    the same brainstorm thread

Write the NEW rolling memory that BOTH characters will read at the start
of their next session, and a structured list of concrete ideas raised in
this session (so a deeper synthesiser can later mine them).

Return ONLY a single JSON object with these keys:

  rolling_summary  string   1500-2400 chars of markdown. WRITE IT FROM
                            A NEUTRAL THIRD-PERSON PERSPECTIVE so each
                            avatar can read it as "what we have explored
                            so far on this thread". Cover:
                              * the brainstorm's evolving topic / focus
                              * the strongest ideas raised so far (most
                                recent + earlier ones still relevant)
                              * unresolved questions / forks worth
                                returning to
                              * any explicit commitments either character
                                made for next time
                            Write so that a character reading it can
                            naturally pick up the thread. Use bullet
                            lists where helpful. Do NOT write in first
                            person, do NOT take sides.
  ideas            array    Each item: { "headline": "5-9 word title",
                                         "detail": "30-90 words of
                                                    concrete content",
                                         "origin": "<character name or
                                                    'human' or 'joint'>" }
                            Aim for 3-8 items. Only include things
                            specific enough to act on.

Hard rules:
- Output a single JSON object. No prose before or after. No markdown
  fences. No comments inside the JSON.
- Use double quotes for strings. Escape newlines inside strings as \\n.
- Keep everything safe-for-work.
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _format_transcript(entries: list[dict]) -> str:
    """Render a list of transcript segments (the same shape we store in
    brainstorm_sessions.transcript_*_json) into a compact 'role: text'
    block, capped at MAX_TRANSCRIPT_CHARS / 2 per side.

    We only keep ``final`` segments — interim deltas would just be
    duplicate prefixes."""
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("final") is False:
            continue
        role = e.get("participantIdentity") or "unknown"
        text = (e.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    out = "\n".join(lines)
    if len(out) > MAX_TRANSCRIPT_CHARS // 2:
        out = out[: MAX_TRANSCRIPT_CHARS // 2] + "\n…[truncated]"
    return out


def _build_user_prompt(
    *,
    avatar_a: dict,
    avatar_b: dict,
    transcript_a: str,
    transcript_b: str,
    prior_memory: Optional[str],
) -> str:
    parts: list[str] = []
    parts.append(
        f"CHARACTER A: {avatar_a.get('character_name', '?')}"
        f"\n  Domain: {(avatar_a.get('domain_summary') or '').strip()}"
    )
    parts.append(
        f"CHARACTER B: {avatar_b.get('character_name', '?')}"
        f"\n  Domain: {(avatar_b.get('domain_summary') or '').strip()}"
    )
    if prior_memory and prior_memory.strip():
        parts.append("PRIOR MEMORY (from earlier sessions on this thread):\n" + prior_memory.strip())
    parts.append(
        "SESSION TRANSCRIPT — view from CHARACTER A's bot.html "
        "(includes A's spoken output and the meeting audio A heard):\n"
        + (transcript_a or "(empty)")
    )
    parts.append(
        "SESSION TRANSCRIPT — view from CHARACTER B's bot.html "
        "(includes B's spoken output and the meeting audio B heard):\n"
        + (transcript_b or "(empty)")
    )
    parts.append("Now produce the JSON object.")
    return "\n\n".join(parts)


def _parse_output(raw: str) -> dict:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in summariser output: {raw[:200]}")
    obj = json.loads(text[start : end + 1])
    summary = obj.get("rolling_summary")
    ideas = obj.get("ideas") or []
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("rolling_summary missing or empty")
    if not isinstance(ideas, list):
        raise ValueError("ideas must be an array")
    return {"rolling_summary": summary.strip(), "ideas": ideas}


def summarise_session(session_id: str) -> dict:
    """Run the summariser on a session and write the result into
    brainstorm_state. Returns the parsed JSON object.
    """
    sess = brainstorm.get_session(session_id)
    if not sess:
        raise LookupError(f"brainstorm session {session_id} not found")
    thread = brainstorm.get_thread(sess["thread_id"])
    if not thread:
        raise LookupError(f"thread {sess['thread_id']} not found")
    avatar_a = storage.get_generation(thread["avatar_a_gen_id"])
    avatar_b = storage.get_generation(thread["avatar_b_gen_id"])
    if not avatar_a or not avatar_b:
        raise LookupError("missing avatar(s) for this thread")

    # Transcripts are stored as JSON-text columns; brainstorm._row_to_dict
    # decodes them into ``transcript_a``/``transcript_b`` keys without the
    # ``_json`` suffix. Fall back to the raw column if needed.
    t_a = sess.get("transcript_a") or sess.get("transcript_a_json") or []
    t_b = sess.get("transcript_b") or sess.get("transcript_b_json") or []
    if isinstance(t_a, str):
        try: t_a = json.loads(t_a)
        except Exception: t_a = []
    if isinstance(t_b, str):
        try: t_b = json.loads(t_b)
        except Exception: t_b = []

    prior_state = brainstorm.get_thread_state(thread["id"]) or {}
    prior_memory = prior_state.get("rolling_summary")

    user_msg = _build_user_prompt(
        avatar_a=avatar_a,
        avatar_b=avatar_b,
        transcript_a=_format_transcript(t_a),
        transcript_b=_format_transcript(t_b),
        prior_memory=prior_memory,
    )

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
            # Trim to fit the personality memory budget — the rolling
            # summary will be injected verbatim into next-session prompts.
            parsed["rolling_summary"] = parsed["rolling_summary"][: prompts.MEMORY_BUDGET]
            brainstorm.write_state(
                thread["id"],
                rolling_summary=parsed["rolling_summary"],
                ideas=parsed["ideas"],
                summariser_model=m,
            )
            with storage._LOCK, storage._connect() as conn:
                conn.execute(
                    "UPDATE brainstorm_sessions SET rolling_summary = ?, status = 'synthesised' WHERE id = ?",
                    (parsed["rolling_summary"], session_id),
                )
            log.info(
                "summariser ok session=%s model=%s ideas=%d summary_chars=%d",
                session_id, m, len(parsed["ideas"]), len(parsed["rolling_summary"]),
            )
            return parsed
        except Exception as e:
            log.warning("summariser model %s failed: %s", m, e)
            last_err = e
            continue
    raise RuntimeError(f"summariser failed across {candidates}: {last_err}")


def summarise_session_safe(session_id: str) -> Optional[dict]:
    """Background-friendly wrapper that swallows exceptions."""
    try:
        return summarise_session(session_id)
    except Exception:
        log.exception("summarise_session_safe failed for %s", session_id)
        return None
