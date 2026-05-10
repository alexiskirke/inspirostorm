"""Build and attach Runway knowledge documents to a generated avatar.

Runway exposes a first-class knowledge-document API
(``client.documents``). Documents are markdown / plain text. They are
attached to an avatar via ``avatars.update(avatar_id,
document_ids=[...])`` and Runway handles chunking + retrieval at
conversation time. Update *replaces* the attached set, so we re-attach
the full list on every refresh.

For each generation we build a "project tour" — a self-contained
markdown bundle the avatar can answer questions from:

  GitHub repo  :  README + repo metadata + recursive file tree
                  + (optionally) the contents of a few key source
                  files / docs, all clearly delimited so retrieval
                  hits coherent sections.

  arXiv paper  :  Title + authors + abstract + extracted PDF text.
                  Section headers are reconstructed from blank-line
                  patterns; large papers are split across multiple
                  documents.

We deliberately cap each individual document around ``MAX_DOC_CHARS``
(default 30 000 chars ≈ 7-8k tokens) and split large sources across
multiple documents instead of one giant blob — this keeps every doc
under whatever per-doc limit Runway enforces internally and gives their
retrieval system multiple named entry points instead of one haystack.
"""
from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from runwayml import RunwayML

from . import github_scan, storage

log = logging.getLogger("scout.knowledge")

MAX_DOC_CHARS = int(os.environ.get("RUNWAY_DOC_MAX_CHARS", "30000"))
MAX_DOCS_PER_AVATAR = int(os.environ.get("RUNWAY_DOC_MAX_DOCS", "6"))
GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _client() -> RunwayML:
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    if not api_key:
        raise RuntimeError(
            "RUNWAYML_API_KEY (or RUNWAYML_API_SECRET) is not set"
        )
    return RunwayML(api_key=api_key)


@dataclass
class DocChunk:
    name: str
    content: str
    bytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.bytes = len(self.content.encode("utf-8"))


def _split_into_chunks(name_prefix: str, body: str, *, max_chars: int = MAX_DOC_CHARS) -> list[DocChunk]:
    """Split ``body`` into <= ``max_chars`` chunks at paragraph boundaries.

    We keep paragraph integrity by splitting on blank lines first and only
    falling back to hard char-cuts when a single paragraph itself exceeds
    the budget.
    """
    body = body.strip()
    if len(body) <= max_chars:
        return [DocChunk(name=name_prefix, content=body)]

    paragraphs = re.split(r"\n\s*\n", body)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Hard-split paragraphs that themselves blow the budget.
        if len(p) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_len = [], 0
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars])
            continue
        if buf_len + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    if len(chunks) == 1:
        return [DocChunk(name=name_prefix, content=chunks[0])]
    return [
        DocChunk(name=f"{name_prefix} (part {i + 1}/{len(chunks)})", content=c)
        for i, c in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# GitHub doc builder
# ---------------------------------------------------------------------------


def _gh_headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_repo_meta(full_name: str) -> dict[str, Any]:
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{full_name}",
            headers=_gh_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return {}


def _gh_file_tree(full_name: str, branch: str | None) -> list[str]:
    """Return up to ~400 paths from the repo's recursive tree."""
    if not branch:
        return []
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{full_name}/git/trees/{branch}",
            headers=_gh_headers(),
            params={"recursive": "1"},
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        tree = data.get("tree", [])
    except requests.RequestException:
        return []
    paths = [
        n["path"]
        for n in tree
        if n.get("type") == "blob" and not _is_noisy_path(n.get("path", ""))
    ]
    return paths[:400]


def _is_noisy_path(path: str) -> bool:
    """Filter out lockfiles, generated assets, third-party blobs, etc."""
    low = path.lower()
    if any(seg in low for seg in (
        "node_modules/", ".git/", ".venv/", "__pycache__/",
        "dist/", "build/", "vendor/",
    )):
        return True
    noisy_suffixes = (
        ".lock", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf",
        ".mp4", ".mov", ".webm", ".wav", ".mp3", ".bin", ".pt", ".onnx",
        ".woff", ".woff2", ".ttf", ".ico", ".svg",
    )
    return low.endswith(noisy_suffixes)


def _gh_file(full_name: str, path: str, *, max_bytes: int = 12_000) -> str:
    """Fetch a single file's raw text. Empty string on failure."""
    headers = _gh_headers()
    headers["Accept"] = "application/vnd.github.raw"
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{full_name}/contents/{path}",
            headers=headers,
            timeout=15,
        )
    except requests.RequestException:
        return ""
    if resp.status_code != 200:
        return ""
    content = resp.text or ""
    if len(content) > max_bytes:
        content = content[:max_bytes] + "\n…[truncated]"
    return content


# Files we always try to include (in priority order) — focused on
# things that explain *what the project does* rather than build glue.
# README is intentionally excluded; it's already embedded in the
# overview document.
# Common source extensions across the languages we care about. Order
# inside the regex doesn't matter; we evaluate patterns in priority
# order below.
_SRC_EXT = r"py|ts|js|tsx|jsx|go|rs|c|cu|cpp|cc|h|hpp|java|kt|swift|rb|cs"

KEY_FILE_PATTERNS = (
    re.compile(r"^docs/.*\.md$", re.IGNORECASE),
    re.compile(r"^(CONTRIBUTING\.md|ARCHITECTURE\.md|DESIGN\.md)$", re.IGNORECASE),
    re.compile(rf"^(main|app|server|index|train|run|cli)\.({_SRC_EXT})$", re.IGNORECASE),
    re.compile(rf"^(train|model|net|inference|infer|eval|sample|generate)_?\w*\.({_SRC_EXT})$", re.IGNORECASE),
    re.compile(rf"^src/(main|app|index|cli|lib|model)\.({_SRC_EXT})$", re.IGNORECASE),
    re.compile(r"^pyproject\.toml$", re.IGNORECASE),
    re.compile(r"^package\.json$", re.IGNORECASE),
    re.compile(r"^Cargo\.toml$", re.IGNORECASE),
)


def _pick_key_files(paths: list[str], *, max_files: int = 4) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for pattern in KEY_FILE_PATTERNS:
        for p in paths:
            if p in seen:
                continue
            if pattern.match(p):
                picked.append(p)
                seen.add(p)
                if len(picked) >= max_files:
                    return picked
    return picked


def build_github_documents(source: dict, *, readme: str = "") -> list[DocChunk]:
    """Produce a list of knowledge documents for a GitHub repo source.

    The first document is the high-signal "tour" (README + metadata +
    file tree). Additional docs are key source files / docs, each on its
    own (so they retrieve cleanly).
    """
    full_name = (source.get("subtitle") or "").strip()
    if not full_name and (source.get("url") or "").startswith("https://github.com/"):
        full_name = source["url"].removeprefix("https://github.com/").strip("/")
    if not full_name:
        return []

    if not readme:
        try:
            readme = github_scan.fetch_readme(full_name, max_chars=MAX_DOC_CHARS)
        except Exception:
            readme = ""

    meta = _gh_repo_meta(full_name)
    branch = meta.get("default_branch")
    tree_paths = _gh_file_tree(full_name, branch)

    repo_block_lines = [
        f"# {meta.get('full_name') or full_name}",
        "",
        meta.get("description") or source.get("description") or "(no description)",
        "",
        "## Repository metadata",
        f"- URL: {meta.get('html_url') or source.get('url') or ''}",
        f"- Default branch: {branch or 'unknown'}",
        f"- Primary language: {meta.get('language') or '(unknown)'}",
        f"- Stars: {meta.get('stargazers_count', 'n/a')}",
        f"- Forks: {meta.get('forks_count', 'n/a')}",
        f"- Topics: {', '.join(meta.get('topics') or []) or '(none)'}",
        f"- License: {(meta.get('license') or {}).get('spdx_id') or '(unspecified)'}",
        f"- Last pushed: {meta.get('pushed_at') or '(unknown)'}",
    ]

    tour_sections: list[str] = ["\n".join(repo_block_lines)]

    if readme.strip():
        tour_sections.append("## README\n\n" + readme.strip())

    if tree_paths:
        tree_text = "\n".join(tree_paths[:200])
        tour_sections.append(f"## File tree (first 200 paths)\n\n```\n{tree_text}\n```")

    tour_md = "\n\n".join(tour_sections)
    docs: list[DocChunk] = _split_into_chunks(f"{full_name} — overview", tour_md)

    # Add up to a handful of key source files as their own documents.
    key_files = _pick_key_files(tree_paths)
    for path in key_files:
        if len(docs) >= MAX_DOCS_PER_AVATAR:
            break
        body = _gh_file(full_name, path)
        if not body.strip():
            continue
        # Wrap source files in a fenced block so retrieval treats them as code.
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        wrapped = f"# {path}\n\nFrom repo `{full_name}`.\n\n```{ext}\n{body}\n```"
        docs.extend(_split_into_chunks(f"{full_name} — {path}", wrapped))

    return docs[:MAX_DOCS_PER_AVATAR]


# ---------------------------------------------------------------------------
# arXiv doc builder
# ---------------------------------------------------------------------------


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Best-effort PDF → text extraction. Returns empty string on failure."""
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; arxiv PDF text will be skipped")
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        out: list[str] = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                out.append(txt)
        return _normalise_pdf_text("\n\n".join(out))
    except Exception as e:
        log.warning("PDF parse failed: %s", e)
        return ""


def _normalise_pdf_text(text: str) -> str:
    """Collapse the worst PDF-extraction artefacts."""
    # PDF extraction often glues hyphenated line-breaks together, leaves
    # form-feeds, and emits triple-spacing. Normalise lightly without
    # destroying paragraph structure.
    text = text.replace("\f", "\n\n")
    text = re.sub(r"-\n(\w)", r"\1", text)            # de-hyphenate broken words
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arxiv_pdf_url(source: dict) -> str | None:
    """Best-effort recovery of the PDF URL for an arXiv source."""
    url = source.get("url") or ""
    if not url:
        return None
    # entry_id usually looks like https://arxiv.org/abs/2401.12345v2
    if "/abs/" in url:
        return url.replace("/abs/", "/pdf/", 1) + ".pdf"
    if url.endswith(".pdf"):
        return url
    return None


def build_arxiv_documents(source: dict) -> list[DocChunk]:
    """Produce a list of knowledge documents for an arXiv paper source."""
    abstract = (source.get("description") or "").strip()
    title = source.get("title") or "Untitled paper"
    authors = source.get("subtitle") or ""

    header = "\n".join(
        [
            f"# {title}",
            f"_{authors}_" if authors else "",
            "",
            "## Abstract",
            "",
            abstract or "(no abstract provided)",
            "",
            f"Source: {source.get('url') or ''}",
        ]
    ).strip()

    docs: list[DocChunk] = [DocChunk(name=f"{title} — abstract", content=header)]

    pdf_url = _arxiv_pdf_url(source)
    if pdf_url:
        try:
            resp = requests.get(pdf_url, timeout=30)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                body_text = _extract_pdf_text(resp.content)
                if body_text:
                    log.info("extracted %d chars from %s", len(body_text), pdf_url)
                    body_md = f"# {title} — full text\n\n{body_text}"
                    docs.extend(_split_into_chunks(f"{title} — full text", body_md))
            else:
                log.info("arxiv pdf %s returned %s", pdf_url, resp.status_code)
        except requests.RequestException as e:
            log.warning("arxiv pdf fetch failed for %s: %s", pdf_url, e)

    return docs[:MAX_DOCS_PER_AVATAR]


# ---------------------------------------------------------------------------
# Orchestration: build docs, upload, attach to avatar
# ---------------------------------------------------------------------------


def build_upload_documents(source: dict) -> list[DocChunk]:
    """Knowledge docs for a user-uploaded zip or pdf source.

    Reads the file from disk (path lives in ``source.meta.upload_path``)
    and produces:

      - For zip: an "overview" doc (title + description + first 200 paths)
        plus per-file docs for whitelisted source files (same key-file
        regex set used for GitHub repos), each fenced with its extension.
      - For pdf: an "overview" doc (title + description) plus chunked
        body text from the parsed PDF.
    """
    meta = source.get("meta") or {}
    kind = meta.get("upload_kind")
    upload_path = meta.get("upload_path")
    if not upload_path:
        return []
    path = Path(upload_path)
    if not path.exists():
        log.warning("upload missing on disk: %s", upload_path)
        return []

    title = source.get("title") or "Custom upload"

    if kind == "zip":
        try:
            data = path.read_bytes()
        except Exception as e:
            log.warning("failed to read zip upload %s: %s", path, e)
            return []
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return []

        # Build a tree of usable text files from the zip.
        all_paths: list[str] = []
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            if _is_zip_path_noisy(n):
                continue
            all_paths.append(n)
        all_paths.sort()

        overview_lines = [
            f"# {title}",
            "",
            (source.get("description") or "(no description)"),
            "",
            "## Source metadata",
            f"- Kind: zip upload",
            f"- Filename: {meta.get('filename') or '(unknown)'}",
            f"- File count: {meta.get('file_count', len(all_paths))}",
            f"- Size: {meta.get('size_bytes', 0)} bytes",
            "",
            "## File tree (first 200 paths)",
            "```",
            "\n".join(all_paths[:200]),
            "```",
        ]
        docs = _split_into_chunks(f"{title} — overview", "\n".join(overview_lines))

        # Pull the same key files we pull from a GitHub repo. Reuse
        # KEY_FILE_PATTERNS by checking each match against the basename
        # AND the full path (zips can have folder prefixes).
        for relpath in _pick_key_files_from_zip(all_paths, max_files=4):
            if len(docs) >= MAX_DOCS_PER_AVATAR:
                break
            try:
                with zf.open(relpath) as fh:
                    body = fh.read(12_000).decode("utf-8", "replace")
            except Exception:
                continue
            if not body.strip():
                continue
            ext = relpath.rsplit(".", 1)[-1] if "." in relpath else ""
            wrapped = (
                f"# {relpath}\n\nFrom upload `{meta.get('filename') or title}`."
                f"\n\n```{ext}\n{body}\n```"
            )
            docs.extend(_split_into_chunks(f"{title} — {relpath}", wrapped))
        return docs[:MAX_DOCS_PER_AVATAR]

    if kind == "pdf":
        try:
            data = path.read_bytes()
        except Exception as e:
            log.warning("failed to read pdf upload %s: %s", path, e)
            return []
        body = _extract_pdf_text(data)
        overview_lines = [
            f"# {title}",
            "",
            (source.get("description") or "(no description)"),
            "",
            f"_Source: PDF upload `{meta.get('filename') or 'document.pdf'}`_",
        ]
        docs: list[DocChunk] = [
            DocChunk(name=f"{title} — overview", content="\n".join(overview_lines))
        ]
        if body:
            docs.extend(_split_into_chunks(f"{title} — full text", body))
        return docs[:MAX_DOCS_PER_AVATAR]

    return []


def _is_zip_path_noisy(path: str) -> bool:
    low = path.lower().rstrip("/")
    if not low or low.startswith("__macosx/"):
        return True
    if any(seg in low for seg in (
        "node_modules/", ".git/", ".venv/", "venv/", "__pycache__/",
        "dist/", "build/", "vendor/", ".idea/", ".vscode/",
    )):
        return True
    return _is_noisy_path(path)


def _pick_key_files_from_zip(paths: list[str], *, max_files: int) -> list[str]:
    """Reuse KEY_FILE_PATTERNS but match against the basename so that
    files inside subdirectories (common in zips of project folders) are
    still found."""
    picked: list[str] = []
    seen: set[str] = set()
    for pattern in KEY_FILE_PATTERNS:
        for p in paths:
            if p in seen:
                continue
            base = p.rsplit("/", 1)[-1]
            if pattern.match(base) or pattern.match(p):
                picked.append(p)
                seen.add(p)
                if len(picked) >= max_files:
                    return picked
    return picked


def build_documents_for(source: dict, *, readme: str = "") -> list[DocChunk]:
    if source.get("source") == "github":
        return build_github_documents(source, readme=readme)
    if source.get("source") == "arxiv":
        return build_arxiv_documents(source)
    if source.get("source") == "upload":
        return build_upload_documents(source)
    return []


def attach_knowledge_for_generation(gen_id: str) -> dict:
    """Build knowledge docs from the generation's source and attach them
    to the existing Runway custom avatar.

    Idempotent: re-running replaces the previously attached doc set.
    """
    storage.update_generation(gen_id, kb_status="building", kb_error=None)
    try:
        rec = storage.get_generation(gen_id)
        if not rec:
            raise LookupError(f"generation {gen_id} not found")
        avatar_id = rec.get("runway_avatar_id")
        if not avatar_id:
            raise RuntimeError(
                "no runway_avatar_id on this generation — create the avatar first"
            )
        source = rec.get("source_meta")
        if not isinstance(source, dict):
            raise RuntimeError("source_meta missing or malformed")

        log.info("building knowledge docs gen=%s source=%s", gen_id, source.get("id"))
        docs = build_documents_for(source, readme=rec.get("readme_excerpt") or "")
        if not docs:
            raise RuntimeError(
                f"no knowledge content could be assembled for source kind "
                f"{source.get('source')!r}"
            )

        client = _client()
        doc_ids: list[str] = []
        total_chars = 0
        for d in docs:
            log.info("uploading doc gen=%s name=%r chars=%d", gen_id, d.name, len(d.content))
            created = client.documents.create(name=d.name[:128], content=d.content)
            did = getattr(created, "id", None) or getattr(created, "document_id", None)
            if not did:
                raise RuntimeError(f"document create returned no id: {created!r}")
            doc_ids.append(did)
            total_chars += len(d.content)

        log.info(
            "attaching %d docs (%d chars total) to avatar=%s",
            len(doc_ids),
            total_chars,
            avatar_id,
        )
        client.avatars.update(avatar_id, document_ids=doc_ids)

        storage.update_generation(
            gen_id,
            runway_document_ids=",".join(doc_ids),
            kb_doc_count=len(doc_ids),
            kb_size_chars=total_chars,
            kb_status="ready",
            kb_error=None,
        )
        log.info("gen=%s knowledge attached (%d docs)", gen_id, len(doc_ids))
        return storage.get_generation(gen_id) or {}
    except Exception as e:
        log.exception("gen=%s knowledge ingestion failed", gen_id)
        storage.update_generation(
            gen_id,
            kb_status="failed",
            kb_error=f"{type(e).__name__}: {e}"[:1000],
        )
        return storage.get_generation(gen_id) or {}
