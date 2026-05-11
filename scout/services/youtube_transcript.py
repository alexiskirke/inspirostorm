"""YouTube-video sources: pull a video's transcript via
youtube-transcript-api, persist the cleaned text, and return a Scout
*source* dict (same shape as github/arxiv/upload/url sources).

We deliberately skip the GPT cleaning step that web URLs go through —
YouTube transcripts are already plain spoken text with no banners or
navigation. We just strip caption-track markers like ``[Music]`` /
``[Applause]`` and join snippets into paragraphs.

The persisted cleaned text lives under
``DATA_DIR/uploads/<source_id>.txt`` alongside other single-item
sources; ``knowledge.build_youtube_documents`` re-reads it when the
user attaches a knowledge base.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from . import storage

log = logging.getLogger("scout.youtube_transcript")

UPLOADS_DIR = storage.DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PREVIEW_CHARS = 6000
OEMBED_TIMEOUT = 8

# Caption-track artefacts we strip before joining (case-insensitive,
# bracketed scene-direction style annotations).
_CAPTION_MARKER_RE = re.compile(r"\[[^\]]{1,40}\]")

# Language preference order: try these in turn before falling back to
# "first available transcript" and finally translation-to-English.
PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]


@dataclass
class StoredYoutube:
    source: dict
    cleaned_path: Path


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Parse any common YouTube URL form (or a bare 11-char ID) into the
    canonical video id. Raises ``ValueError`` on unrecognised inputs."""
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("youtube url is required")
    if _VIDEO_ID_RE.match(s):
        return s

    if not re.match(r"^https?://", s, re.IGNORECASE):
        s = "https://" + s
    try:
        p = urlparse(s)
    except Exception as e:
        raise ValueError(f"could not parse URL: {e}") from e

    host = (p.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host not in {"youtube.com", "youtu.be", "youtube-nocookie.com", "music.youtube.com"}:
        raise ValueError(f"not a youtube URL (host={p.hostname!r})")

    if host == "youtu.be":
        vid = p.path.strip("/").split("/")[0]
    else:
        # /watch?v=ID, /shorts/ID, /embed/ID, /live/ID
        if p.path == "/watch":
            vid = (parse_qs(p.query).get("v") or [""])[0]
        elif p.path.startswith(("/shorts/", "/embed/", "/live/", "/v/")):
            vid = p.path.split("/")[2] if len(p.path.split("/")) > 2 else ""
        else:
            vid = ""

    if not _VIDEO_ID_RE.match(vid):
        raise ValueError(f"could not extract video id from URL ({url_or_id!r})")
    return vid


# ---------------------------------------------------------------------------
# Transcript fetch
# ---------------------------------------------------------------------------


def _fetch_transcript(video_id: str) -> tuple[list, str, bool]:
    """Return ``(snippets, language_code, was_translated)``.

    Strategy:
      1. ``api.fetch(video_id, languages=PREFERRED_LANGUAGES)`` — picks
         the best English track if any (manual or auto).
      2. If no English track exists, enumerate available tracks via
         ``api.list(video_id)``, pick the first one, and translate it
         to English if the track supports translation.
      3. If translation isn't possible (rare), return the native track.
    """
    api = YouTubeTranscriptApi()
    try:
        ft = api.fetch(video_id, languages=PREFERRED_LANGUAGES)
        return list(ft), getattr(ft, "language_code", "en"), False
    except NoTranscriptFound:
        pass  # fall through to translation path

    transcripts = api.list(video_id)
    # The TranscriptList object is iterable over Transcript entries.
    available = list(transcripts)
    if not available:
        raise NoTranscriptFound(video_id, PREFERRED_LANGUAGES, transcripts)

    # Prefer a manually-created track over an auto-generated one; that
    # tends to come out cleaner under translation.
    available.sort(key=lambda t: (getattr(t, "is_generated", True),))
    pick = available[0]
    native_code = getattr(pick, "language_code", "?")

    if getattr(pick, "is_translatable", False):
        log.info(
            "video %s: translating %s -> en",
            video_id, native_code,
        )
        translated = pick.translate("en").fetch()
        return list(translated), "en", True
    log.info(
        "video %s: no english track, no translation available; using native %s",
        video_id, native_code,
    )
    native = pick.fetch()
    return list(native), native_code, False


# ---------------------------------------------------------------------------
# Cleaning / joining
# ---------------------------------------------------------------------------


def _snippets_to_text(snippets: list) -> str:
    """Join transcript snippets into clean prose paragraphs.

    We strip caption-track artefacts like ``[Music]`` / ``[Applause]``
    and collapse small fragment-length snippets into paragraphs of
    roughly sentence-end boundaries (heuristic: insert a paragraph break
    after a snippet that ends in a sentence terminator).
    """
    parts: list[str] = []
    paragraph: list[str] = []
    for s in snippets:
        text = getattr(s, "text", "") or ""
        text = _CAPTION_MARKER_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        paragraph.append(text)
        if text.endswith((".", "!", "?")) and sum(len(p) for p in paragraph) > 180:
            parts.append(" ".join(paragraph))
            paragraph = []
    if paragraph:
        parts.append(" ".join(paragraph))
    body = "\n\n".join(parts).strip()
    return body


# ---------------------------------------------------------------------------
# oEmbed (no API key) for title + author
# ---------------------------------------------------------------------------


def _fetch_oembed(video_id: str) -> dict:
    """Pull title + author_name (channel) without an API key. Empty dict
    on failure — never raises."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=OEMBED_TIMEOUT,
            headers={"User-Agent": "Scout/0.1 (youtube transcript ingest)"},
        )
        if r.status_code == 200:
            return r.json()
    except requests.RequestException as e:
        log.debug("oembed fetch failed for %s: %s", video_id, e)
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_and_store(
    *,
    url: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> StoredYoutube:
    """Parse the URL, pull the transcript, persist it, return a source.

    Raises ``ValueError`` for user-visible errors (bad URL, no transcript,
    captions disabled) and ``RuntimeError`` for unexpected failures.
    """
    try:
        video_id = extract_video_id(url)
    except ValueError:
        raise

    try:
        snippets, lang_code, was_translated = _fetch_transcript(video_id)
    except TranscriptsDisabled:
        raise ValueError("this video has captions disabled")
    except NoTranscriptFound:
        raise ValueError("no transcript available for this video")
    except VideoUnavailable:
        raise ValueError("video is unavailable or private")
    except Exception as e:
        raise RuntimeError(f"transcript fetch failed: {e}") from e

    body = _snippets_to_text(snippets)
    if not body:
        raise ValueError("transcript came back empty after cleaning")

    oembed = _fetch_oembed(video_id)
    channel = oembed.get("author_name") or ""
    youtube_title = oembed.get("title") or ""
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    # Persist alongside other single-item sources for KB ingestion.
    cleaned_path = UPLOADS_DIR / f"yt_{video_id}.txt"
    cleaned_path.write_text(body, encoding="utf-8")
    log.info(
        "stored youtube transcript video=%s lang=%s translated=%s chars=%d -> %s",
        video_id, lang_code, was_translated, len(body), cleaned_path,
    )

    effective_title = (title or "").strip() or youtube_title or f"YouTube video {video_id}"
    preview = body
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS] + "\n\n…[truncated]"

    auto_desc = body[:240].replace("\n", " ").strip()
    if len(body) > 240:
        auto_desc = auto_desc + "…"

    source = {
        "id": f"youtube:{video_id}",
        "source": "youtube",
        "title": effective_title,
        "subtitle": channel or canonical_url,
        "description": (description or "").strip() or auto_desc,
        "url": canonical_url,
        "meta": {
            "video_id": video_id,
            "channel": channel,
            "language_code": lang_code,
            "was_translated": was_translated,
            "snippet_count": len(snippets),
            "cleaned_path": str(cleaned_path),
            "cleaned_chars": len(body),
        },
        "preview_text": preview,
    }
    return StoredYoutube(source=source, cleaned_path=cleaned_path)
