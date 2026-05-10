"""Custom-source uploads: persist a user-supplied zip / pdf / folder
and convert it into a Scout *source* dict (the same shape returned by
``github_scan.fetch_repos`` and ``arxiv_scan.fetch_papers``).

The persisted file lives under ``DATA_DIR/uploads/<source_id>.<ext>``;
only the lightweight source dict (with title, description, on-disk path
and a short text "preview") is returned to the frontend. The full
ingestion — walking a zip into per-file documents, splitting a PDF into
chunks — happens later in ``knowledge.build_upload_documents`` when the
user actually clicks *Generate avatars* on the new card.
"""
from __future__ import annotations

import io
import logging
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import storage

log = logging.getLogger("scout.uploads")

UPLOADS_DIR = storage.DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Hard caps to protect the server. We only want hackathon-scale stuff.
MAX_UPLOAD_BYTES = int(50 * 1024 * 1024)  # 50 MB; bigger needs streaming
PREVIEW_CHARS = 6000                       # text we feed to identity-gen


# ---------------------------------------------------------------------------


@dataclass
class StoredUpload:
    """Persisted upload + the lightweight source dict for the gallery."""

    source: dict
    upload_path: Path


# ---------------------------------------------------------------------------
# File-tree helpers (zip)
# ---------------------------------------------------------------------------


def _is_noisy_zip_path(path: str) -> bool:
    """Same filter as ``knowledge._is_noisy_path`` but lighter — zip uploads
    often have no .git but can have everything else."""
    low = path.lower().rstrip("/")
    if not low or low.startswith("__macosx/"):
        return True
    if any(seg in low for seg in (
        "node_modules/", ".git/", ".venv/", "venv/", "__pycache__/",
        "dist/", "build/", "vendor/", ".idea/", ".vscode/",
    )):
        return True
    noisy_suffixes = (
        ".lock", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf",
        ".mp4", ".mov", ".webm", ".wav", ".mp3", ".bin", ".pt", ".onnx",
        ".woff", ".woff2", ".ttf", ".ico", ".svg", ".zip", ".tar",
        ".gz", ".bz2", ".7z", ".rar", ".dmg", ".exe", ".dll", ".so",
        ".pyc", ".pyo", ".class",
    )
    return low.endswith(noisy_suffixes)


def _zip_preview(zip_bytes: bytes) -> tuple[str, list[str]]:
    """Return ``(preview_text, file_tree)`` for the identity prompt.

    Preview text = README contents (if any) + a sampled file listing.
    Truncated to PREVIEW_CHARS so the LLM call stays cheap.
    """
    preview_parts: list[str] = []
    tree: list[str] = []
    readme_text = ""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            for n in names:
                if n.endswith("/"):
                    continue
                if _is_noisy_zip_path(n):
                    continue
                tree.append(n)
                # Find best README candidate (top-level wins).
                low = n.lower().rsplit("/", 1)[-1]
                if low in ("readme.md", "readme.txt", "readme") and not readme_text:
                    try:
                        with zf.open(n) as fh:
                            readme_text = fh.read(20000).decode("utf-8", "replace")
                    except Exception:
                        pass
            tree.sort()
    except zipfile.BadZipFile:
        return ("[upload was not a valid zip]", [])

    if readme_text.strip():
        preview_parts.append("# README\n" + readme_text.strip())
    if tree:
        sample = tree[:120]
        preview_parts.append("# File listing (first 120 paths)\n" + "\n".join(sample))

    preview = "\n\n".join(preview_parts)
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS] + "\n\n…[truncated]"
    return preview, tree


def _pdf_preview(pdf_bytes: bytes) -> str:
    """Extract the first PREVIEW_CHARS of plain text from a PDF for the
    identity prompt. Empty string if extraction fails."""
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; PDF preview will be empty")
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        log.warning("PDF parse failed: %s", e)
        return ""
    out: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt:
            continue
        out.append(txt)
        total += len(txt)
        if total >= PREVIEW_CHARS * 2:
            break
    text = _normalise_pdf_text("\n\n".join(out))
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + "\n\n…[truncated]"
    return text


def _normalise_pdf_text(text: str) -> str:
    text = text.replace("\f", "\n\n")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_upload(
    *,
    title: str,
    description: str,
    filename: str,
    content: bytes,
) -> StoredUpload:
    """Persist a single uploaded file and produce the source dict.

    Raises ``ValueError`` for unsupported file types or oversized inputs.
    """
    if not title.strip():
        raise ValueError("title is required")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"upload too big: {len(content)} bytes (max {MAX_UPLOAD_BYTES})"
        )

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "zip":
        kind = "zip"
        preview, tree = _zip_preview(content)
        meta_extra = {"file_count": len(tree)}
    elif ext == "pdf":
        kind = "pdf"
        preview = _pdf_preview(content)
        meta_extra = {"text_chars": len(preview)}
    else:
        raise ValueError(
            f"unsupported file type {ext!r}; only .zip and .pdf are accepted"
        )

    source_id = f"upload:{uuid.uuid4().hex}"
    on_disk_name = f"{source_id.split(':',1)[1]}.{kind}"
    upload_path = UPLOADS_DIR / on_disk_name
    upload_path.write_bytes(content)
    log.info(
        "stored upload kind=%s name=%r size=%d -> %s",
        kind, filename, len(content), upload_path,
    )

    source = {
        "id": source_id,
        "source": "upload",
        "title": title.strip(),
        "subtitle": filename,
        "description": (description or "").strip() or "(no description provided)",
        "url": None,
        "meta": {
            "upload_kind": kind,
            "upload_path": str(upload_path),
            "filename": filename,
            "size_bytes": len(content),
            **meta_extra,
        },
        # ``preview_text`` is consumed by main._readme_for() and by
        # knowledge.build_upload_documents — kept on the source dict so
        # both code paths can reach it without re-reading the file.
        "preview_text": preview,
    }
    return StoredUpload(source=source, upload_path=upload_path)
