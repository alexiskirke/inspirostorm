"""Persistent storage for generations: SQLite metadata + file-on-disk images.

Both live under ``DATA_DIR`` (defaults to ``<repo>/scout/data`` locally; on
Railway, point ``DATA_DIR`` at the mounted volume, e.g. ``/data``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


DATA_DIR = Path(os.environ.get("DATA_DIR") or _default_data_dir()).resolve()
IMAGES_DIR = DATA_DIR / "images"
MOVIES_DIR = DATA_DIR / "movies"
DB_PATH = DATA_DIR / "scout.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MOVIES_DIR.mkdir(parents=True, exist_ok=True)


_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _LOCK, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id              TEXT PRIMARY KEY,
                source_id       TEXT NOT NULL,
                source_type     TEXT NOT NULL,
                source_title    TEXT,
                source_url      TEXT,
                source_meta     TEXT,
                prompt          TEXT NOT NULL,
                model           TEXT NOT NULL,
                ratio           TEXT NOT NULL,
                runway_task_id  TEXT,
                image_path      TEXT,
                status          TEXT NOT NULL,
                failure_reason  TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_gen_source     ON generations(source_id);
            CREATE INDEX IF NOT EXISTS idx_gen_created_at ON generations(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_gen_status     ON generations(status);

            -- One thread per ordered pair of avatars (a,b). The pair_key
            -- collapses (a,b) and (b,a) so calling start_session with the
            -- avatars in either order resolves to the same thread.
            CREATE TABLE IF NOT EXISTS brainstorm_threads (
                id                 TEXT PRIMARY KEY,
                pair_key           TEXT NOT NULL UNIQUE,
                avatar_a_gen_id    TEXT NOT NULL,
                avatar_b_gen_id    TEXT NOT NULL,
                topic_seed         TEXT,
                created_at         TEXT NOT NULL,
                last_session_at    TEXT,
                status             TEXT NOT NULL DEFAULT 'active'
            );
            CREATE INDEX IF NOT EXISTS idx_thread_pair    ON brainstorm_threads(pair_key);
            CREATE INDEX IF NOT EXISTS idx_thread_recent  ON brainstorm_threads(last_session_at DESC);

            -- Each brainstorm session = one meeting where the pair talked.
            -- Two Runway+meet sessions (one per avatar) live behind a single row.
            CREATE TABLE IF NOT EXISTS brainstorm_sessions (
                id                 TEXT PRIMARY KEY,
                thread_id          TEXT NOT NULL,
                topic              TEXT,
                meeting_url        TEXT,
                meet_session_id_a  TEXT,
                meet_session_id_b  TEXT,
                runway_session_id_a TEXT,
                runway_session_id_b TEXT,
                started_at         TEXT NOT NULL,
                ended_at           TEXT,
                end_reason         TEXT,
                transcript_a_json  TEXT,
                transcript_b_json  TEXT,
                rolling_summary    TEXT,
                synthesis_id       TEXT,
                status             TEXT NOT NULL DEFAULT 'live'
            );
            CREATE INDEX IF NOT EXISTS idx_brsess_thread  ON brainstorm_sessions(thread_id, started_at DESC);

            -- Per-thread rolling memory blob (one row per thread). Used to
            -- inject "ongoing brainstorm" context into both avatars'
            -- personality on the next session.
            CREATE TABLE IF NOT EXISTS brainstorm_state (
                thread_id          TEXT PRIMARY KEY,
                rolling_summary    TEXT,
                ideas_json         TEXT,
                summariser_model   TEXT,
                updated_at         TEXT NOT NULL,
                version            INTEGER NOT NULL DEFAULT 0
            );

            -- Big creative outputs (gpt-5.4 thread synthesis + movie pitch).
            -- Multiple per thread are allowed (re-syntheses, refinements).
            CREATE TABLE IF NOT EXISTS brainstorm_synthesis (
                id                 TEXT PRIMARY KEY,
                thread_id          TEXT NOT NULL,
                scope              TEXT NOT NULL,           -- 'session' | 'thread'
                source_session_id  TEXT,
                text_md            TEXT NOT NULL,
                movie_pitch        TEXT,
                ideas_json         TEXT,
                model_used         TEXT,
                created_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_syn_thread     ON brainstorm_synthesis(thread_id, created_at DESC);
            """
        )
        # Idempotent additions for new movie-generation columns (Phase 7).
        _ensure_columns(
            conn,
            "brainstorm_synthesis",
            {
                "movie_status":           "TEXT",     # 'idle' | 'building' | 'ready' | 'failed'
                "movie_error":            "TEXT",
                "movie_path":             "TEXT",     # filename under DATA_DIR/movies/
                "movie_prompt":           "TEXT",     # what we actually sent to Runway
                "movie_model":            "TEXT",     # 'gen3a_turbo' | 'veo3.1_fast' | ...
                "movie_runway_task_id":   "TEXT",
                "movie_created_at":       "TEXT",
            },
        )
        _ensure_columns(
            conn,
            "generations",
            {
                "character_name":      "TEXT",
                "personality":         "TEXT",
                "start_script":        "TEXT",
                "voice_preset":        "TEXT",
                "readme_excerpt":      "TEXT",
                "avatar_status":       "TEXT",
                "avatar_error":        "TEXT",
                "runway_avatar_id":    "TEXT",
                "avatar_created_at":   "TEXT",
                "kb_status":           "TEXT",
                "kb_error":            "TEXT",
                "kb_doc_count":        "INTEGER",
                "kb_size_chars":       "INTEGER",
                "runway_document_ids": "TEXT",  # comma-separated

                # Persona v2 fields (Phase 1 of brainstorming roadmap):
                # `domain_body` is the LLM-emitted domain section that
                # gets composed into `personality` at compose time.
                # `domain_summary` is the third-person briefing other
                # avatars see when paired with this one.
                # `weirdness` 0.0..1.0 modulates the OPERATING MODE preamble.
                "domain_body":         "TEXT",
                "domain_summary":      "TEXT",
                "weirdness":           "REAL",
            },
        )


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Add any missing columns idempotently (lightweight migration)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_generation(
    *,
    source: dict,
    prompt: str,
    model: str,
    ratio: str,
    identity: Optional[dict] = None,
    readme_excerpt: str = "",
) -> str:
    """Insert a 'pending' generation, return its new id.

    ``identity`` is the full LLM-generated persona package
    (image_prompt, character_name, personality, start_script,
    voice_preset). It's optional only because the legacy callers may not
    have it yet; new callers should always pass it.
    """
    gen_id = uuid.uuid4().hex
    identity = identity or {}
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO generations (
                id, source_id, source_type, source_title, source_url, source_meta,
                prompt, model, ratio, status, created_at,
                character_name, personality, start_script, voice_preset, readme_excerpt,
                domain_body, domain_summary, weirdness
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                gen_id,
                source["id"],
                source["source"],
                source.get("title"),
                source.get("url"),
                json.dumps(source, default=str),
                prompt,
                model,
                ratio,
                "pending",
                _now(),
                identity.get("character_name"),
                identity.get("personality"),
                identity.get("start_script"),
                identity.get("voice_preset"),
                readme_excerpt[:6000] if readme_excerpt else None,
                identity.get("domain_body"),
                identity.get("domain_summary"),
                identity.get("weirdness"),
            ),
        )
    return gen_id


def update_generation(gen_id: str, **fields: Any) -> None:
    if not fields:
        return
    keys = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [gen_id]
    with _LOCK, _connect() as conn:
        conn.execute(f"UPDATE generations SET {keys} WHERE id = ?", values)


def mark_running(gen_id: str, runway_task_id: str) -> None:
    update_generation(gen_id, runway_task_id=runway_task_id, status="running")


def mark_succeeded(gen_id: str, image_path: str) -> None:
    update_generation(
        gen_id,
        status="succeeded",
        image_path=image_path,
        completed_at=_now(),
    )


def mark_failed(gen_id: str, reason: str) -> None:
    update_generation(
        gen_id,
        status="failed",
        failure_reason=reason[:1000],
        completed_at=_now(),
    )


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("source_meta"):
        try:
            d["source_meta"] = json.loads(d["source_meta"])
        except json.JSONDecodeError:
            pass
    if d.get("image_path"):
        d["image_url"] = f"/data/images/{d['image_path']}"
    return d


def get_generation(gen_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM generations WHERE id = ?", (gen_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_generations(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM generations WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    by_id = {r["id"]: _row_to_dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def list_generations(limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM generations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_generations_for_source(source_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM generations WHERE source_id = ? "
            "ORDER BY created_at DESC",
            (source_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def save_image_bytes(gen_id: str, content: bytes, ext: str = "png") -> str:
    """Write image bytes to disk and return the relative filename."""
    safe_ext = ext.lstrip(".").lower() or "png"
    filename = f"{gen_id}.{safe_ext}"
    (IMAGES_DIR / filename).write_bytes(content)
    return filename


def reset_brainstorm() -> dict:
    """Wipe ALL brainstorm history: threads, sessions, rolling memory,
    syntheses, and the per-synthesis movie + composite files on disk.

    Does NOT touch the ``generations`` table (the avatars themselves —
    those took GPT calls and Runway credits to make and the user almost
    certainly wants to keep them). Does NOT touch the Runway-side
    custom avatars or attached documents.

    Idempotent. Returns a dict of how many rows / files were removed.
    Safe to call mid-build: in-flight movie pipelines will see their
    synthesis row vanish and their final UPDATE will be a no-op.
    """
    counts = {
        "threads": 0,
        "sessions": 0,
        "state_rows": 0,
        "syntheses": 0,
        "movies_deleted": 0,
        "composites_deleted": 0,
    }
    with _LOCK, _connect() as conn:
        for table, key in [
            ("brainstorm_threads",   "threads"),
            ("brainstorm_sessions",  "sessions"),
            ("brainstorm_state",     "state_rows"),
            ("brainstorm_synthesis", "syntheses"),
        ]:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[key] = row["n"] if row else 0
            conn.execute(f"DELETE FROM {table}")
    # Per-synthesis files on disk. Every .mp4 in MOVIES_DIR is a synthesis
    # output, and every .png in DATA_DIR/composites is a clip composite.
    composites_dir = DATA_DIR / "composites"
    for d, ext, key in [
        (MOVIES_DIR, ".mp4", "movies_deleted"),
        (composites_dir, ".png", "composites_deleted"),
    ]:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() == ext:
                try:
                    f.unlink()
                    counts[key] += 1
                except Exception:
                    pass
    return counts


def live_brainstorm_sessions_for_avatar(gen_id: str) -> list[str]:
    """Return session ids that are currently ``status='live'`` and have
    this avatar in either slot. Used by the delete endpoint to refuse
    teardown while a brainstorm is mid-flight."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id
              FROM brainstorm_sessions s
              JOIN brainstorm_threads t ON t.id = s.thread_id
             WHERE s.status = 'live'
               AND (t.avatar_a_gen_id = ? OR t.avatar_b_gen_id = ?)
            """,
            (gen_id, gen_id),
        ).fetchall()
    return [r["id"] for r in rows]


def delete_generation(gen_id: str) -> Optional[dict]:
    """Remove a generation row + its on-disk image file.

    Returns the deleted record (in the same shape as ``get_generation``)
    or ``None`` if no row with that id existed. Brainstorm threads /
    sessions / syntheses that reference this avatar are LEFT INTACT —
    they'll have a dangling reference, which the UI handles by showing
    the missing participant as "(avatar deleted)".

    Runway-side artifacts (custom character, attached documents) are
    NOT cleaned up here — call ``avatars.delete_runway_artifacts``
    first if you want a full teardown.
    """
    rec = get_generation(gen_id)
    if not rec:
        return None
    image_path = rec.get("image_path")
    if image_path:
        file = IMAGES_DIR / image_path
        try:
            file.unlink(missing_ok=True)
        except Exception:
            pass
    with _LOCK, _connect() as conn:
        conn.execute("DELETE FROM generations WHERE id = ?", (gen_id,))
    return rec
