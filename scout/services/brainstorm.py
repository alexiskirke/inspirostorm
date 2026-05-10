"""Brainstorm orchestration: pair two of our custom avatars, dispatch
them into a meeting, and remember the conversation across sessions.

Key entities (see ``storage.init_db`` for the SQLite schema):

  brainstorm_threads      one per ordered pair (collapsed by pair_key)
  brainstorm_sessions     one per actual meeting; carries the two
                          per-avatar Recall bot ids and Runway session
                          ids, plus the captured transcripts
  brainstorm_state        rolling per-thread memory blob (gpt-5.1
                          summariser writes here on session end)
  brainstorm_synthesis    deeper gpt-5.5 outputs + movie pitch

Lifecycle (happy path):

  start_session(...)
      composes per-avatar personality
        = OPERATING_MODE
        + DOMAIN BODY
        + PARTNER briefing (built from the *other* avatar's
          domain_summary)
        + ONGOING brainstorm memory (read from brainstorm_state)
      POSTs both avatars to the meet server's /api/start with that
      personality override; stores session row.

  poll_session(...) (optional)
      pulls live transcripts + bot status; the same data is also
      available directly from the meet server.

  end_session(..., reason)
      tells the meet server to stop both bots, snapshots the
      transcripts into the row, then calls summarise_into_state() to
      let the next session pick up where this one left off. Synthesis
      (gpt-5.5) is intentionally separate so it can be re-run on
      demand without touching the per-session memory plumbing.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from pathlib import Path

from dotenv import dotenv_values

from . import prompts, storage

log = logging.getLogger("scout.brainstorm")

MEET_BASE_URL = os.environ.get("MEET_BASE_URL", "http://localhost:3000")
DEFAULT_MEETING_URL = os.environ.get("BRAINSTORM_DEFAULT_MEETING_URL", "")

# Path to the upstream runway-characters-meet .env (cloned to ./meet).
# We peek at PUBLIC_URL there so we can pre-flight the tunnel before
# letting Recall.ai be told to spawn bots that will inevitably fail.
_MEET_ENV_PATH = Path(__file__).resolve().parents[2] / "meet" / ".env"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_key(a: str, b: str) -> str:
    """Canonical key so (A,B) and (B,A) collapse to the same thread."""
    return ":".join(sorted([a, b]))


def _row_to_dict(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    d = dict(row)
    for key in ("transcript_a_json", "transcript_b_json", "ideas_json"):
        if d.get(key):
            try:
                d[key.removesuffix("_json") if key.endswith("_json") else key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return d


def _load_avatar(gen_id: str) -> dict:
    """Fetch a custom avatar generation row and validate it's
    Brainstorm-ready (has runway_avatar_id + persona v2 fields)."""
    rec = storage.get_generation(gen_id)
    if not rec:
        raise LookupError(f"avatar generation {gen_id} not found")
    missing: list[str] = []
    if not rec.get("runway_avatar_id"):
        missing.append("runway_avatar_id (run 'Make this an avatar' first)")
    for key in ("domain_body", "domain_summary", "voice_preset"):
        if not rec.get(key):
            missing.append(f"{key} (re-run scout/scripts/backfill_persona_v2.py)")
    if missing:
        raise RuntimeError(
            f"avatar {gen_id!r} not Brainstorm-ready, missing: {', '.join(missing)}"
        )
    return rec


def _runway_key() -> str:
    k = os.environ.get("RUNWAYML_API_KEY") or os.environ.get("RUNWAYML_API_SECRET")
    if not k:
        raise RuntimeError("RUNWAYML_API_KEY is not set")
    return k


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def find_or_create_thread(
    avatar_a_gen_id: str,
    avatar_b_gen_id: str,
    *,
    topic_seed: Optional[str] = None,
) -> dict:
    """Return the brainstorm_thread row for (a,b), creating it if absent.
    Order-insensitive — the canonical pair_key collapses (A,B) and (B,A)."""
    if avatar_a_gen_id == avatar_b_gen_id:
        raise ValueError("brainstorm requires two DIFFERENT avatars")

    pair_key = _pair_key(avatar_a_gen_id, avatar_b_gen_id)
    with storage._LOCK, storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM brainstorm_threads WHERE pair_key = ?", (pair_key,)
        ).fetchone()
        if row:
            d = dict(row)
            # If a new topic_seed is supplied, update it (rare; for now
            # treat the latest as authoritative).
            if topic_seed and topic_seed.strip() and topic_seed.strip() != (d.get("topic_seed") or ""):
                conn.execute(
                    "UPDATE brainstorm_threads SET topic_seed = ? WHERE id = ?",
                    (topic_seed.strip(), d["id"]),
                )
                d["topic_seed"] = topic_seed.strip()
            return d
        thread_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO brainstorm_threads (
                id, pair_key, avatar_a_gen_id, avatar_b_gen_id,
                topic_seed, created_at, status
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                thread_id,
                pair_key,
                avatar_a_gen_id,
                avatar_b_gen_id,
                topic_seed.strip() if topic_seed else None,
                _now(),
                "active",
            ),
        )
        log.info(
            "created brainstorm thread %s for pair %s",
            thread_id,
            pair_key,
        )
    return get_thread(thread_id) or {}


def get_thread(thread_id: str) -> Optional[dict]:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM brainstorm_threads WHERE id = ?", (thread_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_threads() -> list[dict]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT t.*,
                   ga.character_name AS avatar_a_name,
                   ga.image_path     AS avatar_a_image,
                   gb.character_name AS avatar_b_name,
                   gb.image_path     AS avatar_b_image,
                   (SELECT COUNT(*) FROM brainstorm_sessions s
                     WHERE s.thread_id = t.id) AS session_count
            FROM brainstorm_threads t
            JOIN generations ga ON ga.id = t.avatar_a_gen_id
            JOIN generations gb ON gb.id = t.avatar_b_gen_id
            ORDER BY COALESCE(t.last_session_at, t.created_at) DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_thread_state(thread_id: str) -> Optional[dict]:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM brainstorm_state WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_sessions(thread_id: str) -> list[dict]:
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM brainstorm_sessions WHERE thread_id = ? "
            "ORDER BY started_at DESC",
            (thread_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Personality composition for a session
# ---------------------------------------------------------------------------


def _compose_session_personality(
    self_avatar: dict,
    partner_avatar: dict,
    rolling_memory: Optional[str],
    session_brief: Optional[str] = None,
) -> str:
    """Build the per-session personality override for ``self_avatar``.

    Layout produced by ``prompts.compose_personality``:
        SESSION BRIEF   (per-session task; overrides everything; defuses
                         "how can I help you" default first-turn behavior)
        OPERATING MODE  (with the avatar's own weirdness dial)
        DOMAIN          (the avatar's domain_body)
        PARTNER         (briefing built from partner's domain_summary)
        ONGOING         (rolling brainstorm memory, optional)
    """
    return prompts.compose_personality(
        domain_body=self_avatar["domain_body"],
        weirdness=float(self_avatar.get("weirdness") or 0.33),
        partner_name=partner_avatar.get("character_name"),
        partner_summary=partner_avatar.get("domain_summary"),
        brainstorm_memory=rolling_memory,
        session_brief=session_brief,
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def start_session(
    thread_id: str,
    *,
    meeting_url: Optional[str] = None,
    topic: Optional[str] = None,
) -> dict:
    """Dispatch both avatars of ``thread_id`` into a meeting via the meet
    server. Returns the brainstorm_sessions row.

    ``topic`` overrides the thread's stored topic_seed for this session
    only (e.g. *"today let's focus on the dataloader"*).
    """
    thread = get_thread(thread_id)
    if not thread:
        raise LookupError(f"brainstorm thread {thread_id} not found")
    avatar_a = _load_avatar(thread["avatar_a_gen_id"])
    avatar_b = _load_avatar(thread["avatar_b_gen_id"])
    state = get_thread_state(thread_id) or {}
    rolling = state.get("rolling_summary")

    meeting = (meeting_url or DEFAULT_MEETING_URL or "").strip()
    if not meeting:
        raise ValueError(
            "no meeting_url provided and BRAINSTORM_DEFAULT_MEETING_URL is unset"
        )

    effective_topic = (
        topic.strip() if (topic and topic.strip())
        else (thread.get("topic_seed") or "").strip()
    )

    # Per-avatar overrides. Each avatar gets a personality where THEIR
    # SESSION BRIEF + OPERATING MODE + DOMAIN sit at the top, the OTHER
    # avatar appears in the PARTNER block, and the rolling brainstorm
    # memory (if any) is appended.
    brief = effective_topic or None
    pers_a = _compose_session_personality(avatar_a, avatar_b, rolling, session_brief=brief)
    pers_b = _compose_session_personality(avatar_b, avatar_a, rolling, session_brief=brief)

    # Per-avatar startScript: framed as a brainstorm OPENER (an angle
    # from the avatar's own domain) rather than "what brings you here".
    # We deliberately do NOT use the stored stock start_script when
    # there's no topic — older personas' stock openers were written in
    # service-mode (Sage's "Tell me what meditation you're making…")
    # which collapses into 1-on-1 coaching. A generic "what angle do
    # you bring" is safer until we re-backfill those personas.
    if effective_topic:
        start_a = (
            f"Hi {avatar_b.get('character_name', 'there')} — today's brainstorm: "
            f"{effective_topic}. Here's one angle from my side to kick off: "
        )
        start_b = (
            f"Hi {avatar_a.get('character_name', 'there')} — and the angle from "
            f"my side on \"{effective_topic}\" is: "
        )
    else:
        start_a = (
            f"Hi {avatar_b.get('character_name', 'there')} — let's brainstorm "
            f"something at the intersection of our domains. Here's one angle from my side: "
        )
        start_b = (
            f"Hi {avatar_a.get('character_name', 'there')} — and from mine, here's one angle: "
        )

    runway_key = _runway_key()

    # Pre-flight: make sure the meet server is reachable AND its public
    # tunnel resolves. If we don't check, Recall.ai cheerfully tries to
    # join the Zoom and then can't fetch bot.html, leaving zombie tiles
    # in the meeting (we hit this twice today — see tunnel_watchdog.py
    # for the auto-fix).
    try:
        meet_health = requests.get(f"{MEET_BASE_URL}/", timeout=5)
        if meet_health.status_code >= 500:
            raise RuntimeError(f"meet returned {meet_health.status_code}")
    except requests.RequestException as e:
        raise RuntimeError(
            f"meet server is not reachable at {MEET_BASE_URL} ({e}). "
            "Run scout/scripts/tunnel_watchdog.py to bring it up + keep "
            "the cloudflared tunnel healthy."
        )
    # Also probe the PUBLIC_URL out of meet/.env — that's the URL
    # Recall.ai will actually fetch bot.html from. NXDOMAIN here means
    # Cloudflare has revoked the trycloudflare quick-tunnel.
    pub = ""
    if _MEET_ENV_PATH.exists():
        pub = (dotenv_values(_MEET_ENV_PATH).get("PUBLIC_URL") or "").strip()
    if pub and pub.startswith(("http://", "https://")):
        try:
            r = requests.head(pub, timeout=5, allow_redirects=False)
            if r.status_code >= 500:
                raise RuntimeError(f"tunnel HTTP {r.status_code}")
        except requests.RequestException as e:
            raise RuntimeError(
                f"meet's public tunnel ({pub}) is unreachable ({e}). "
                "The cloudflared quick-tunnel has likely been evicted by "
                "Cloudflare. Restart scout/scripts/tunnel_watchdog.py to "
                "rotate it (the watchdog does this automatically once it's "
                "running)."
            )

    payloads = [
        ("a", avatar_a, pers_a, start_a),
        ("b", avatar_b, pers_b, start_b),
    ]
    sess_ids: dict[str, str] = {}
    for slot, avatar, personality, start_script in payloads:
        body = {
            "meetingUrl": meeting,
            "avatarType": "custom",
            "avatarId": avatar["runway_avatar_id"],
            "botName": avatar.get("character_name") or "Brainstorm partner",
            "personality": personality,
            "startScript": start_script,
        }
        log.info(
            "dispatching brainstorm avatar slot=%s name=%r personality_chars=%d",
            slot, body["botName"], len(personality),
        )
        r = requests.post(
            f"{MEET_BASE_URL}/api/start",
            headers={
                "Content-Type": "application/json",
                "X-Runway-Key": runway_key,
            },
            json=body,
            timeout=20,
        )
        r.raise_for_status()
        sess_ids[slot] = r.json()["sessionId"]

    session_id = uuid.uuid4().hex
    with storage._LOCK, storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO brainstorm_sessions (
                id, thread_id, topic, meeting_url,
                meet_session_id_a, meet_session_id_b,
                started_at, status
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                session_id, thread_id, effective_topic or None, meeting,
                sess_ids["a"], sess_ids["b"], _now(), "live",
            ),
        )
        conn.execute(
            "UPDATE brainstorm_threads SET last_session_at = ? WHERE id = ?",
            (_now(), thread_id),
        )
    return get_session(session_id) or {}


def get_session(session_id: str) -> Optional[dict]:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM brainstorm_sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return _row_to_dict(row)


def end_session(session_id: str, *, reason: str = "manual") -> dict:
    """Stop both avatars on the meet server, snapshot transcripts, and
    trigger summarisation. Idempotent — safe to call twice."""
    sess = get_session(session_id)
    if not sess:
        raise LookupError(f"brainstorm session {session_id} not found")
    if sess.get("status") in {"ended", "synthesised"}:
        return sess

    transcripts: dict[str, list] = {"a": [], "b": []}
    for slot in ("a", "b"):
        meet_id = sess.get(f"meet_session_id_{slot}")
        if not meet_id:
            continue
        try:
            r = requests.post(
                f"{MEET_BASE_URL}/api/sessions/{meet_id}/stop", timeout=10
            )
            log.info("meet stop slot=%s status=%s", slot, r.status_code)
        except requests.RequestException as e:
            log.warning("meet stop failed slot=%s: %s", slot, e)
        try:
            r = requests.get(
                f"{MEET_BASE_URL}/api/sessions/{meet_id}/transcript", timeout=10
            )
            if r.ok:
                transcripts[slot] = r.json().get("entries", [])
        except requests.RequestException as e:
            log.warning("meet transcript fetch failed slot=%s: %s", slot, e)

    storage._connect().close()  # ensure WAL flushed before next write
    with storage._LOCK, storage._connect() as conn:
        conn.execute(
            """
            UPDATE brainstorm_sessions
               SET ended_at = ?, end_reason = ?,
                   transcript_a_json = ?, transcript_b_json = ?,
                   status = 'ended'
             WHERE id = ?
            """,
            (
                _now(), reason,
                json.dumps(transcripts["a"]),
                json.dumps(transcripts["b"]),
                session_id,
            ),
        )
    log.info(
        "session %s ended reason=%s transcript_lens=(%d,%d)",
        session_id, reason, len(transcripts["a"]), len(transcripts["b"]),
    )
    return get_session(session_id) or {}


# ---------------------------------------------------------------------------
# Summariser & state writer (Phase 5 will plug in gpt-5.1 here)
# ---------------------------------------------------------------------------


def write_state(
    thread_id: str,
    *,
    rolling_summary: str,
    ideas: Optional[list[Any]] = None,
    summariser_model: Optional[str] = None,
) -> None:
    """Upsert the per-thread brainstorm_state row. Phase 5 (the
    gpt-5.1 summariser) calls this after every session_end."""
    payload = {
        "thread_id": thread_id,
        "rolling_summary": rolling_summary,
        "ideas_json": json.dumps(ideas) if ideas is not None else None,
        "summariser_model": summariser_model,
        "updated_at": _now(),
    }
    with storage._LOCK, storage._connect() as conn:
        existing = conn.execute(
            "SELECT version FROM brainstorm_state WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE brainstorm_state
                   SET rolling_summary = :rolling_summary,
                       ideas_json = :ideas_json,
                       summariser_model = :summariser_model,
                       updated_at = :updated_at,
                       version = version + 1
                 WHERE thread_id = :thread_id
                """,
                payload,
            )
        else:
            payload["version"] = 1
            conn.execute(
                """
                INSERT INTO brainstorm_state (
                    thread_id, rolling_summary, ideas_json,
                    summariser_model, updated_at, version
                ) VALUES (
                    :thread_id, :rolling_summary, :ideas_json,
                    :summariser_model, :updated_at, :version
                )
                """,
                payload,
            )
