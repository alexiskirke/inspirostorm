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
DB_PATH = DATA_DIR / "scout.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


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
            """
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
                character_name, personality, start_script, voice_preset, readme_excerpt
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
