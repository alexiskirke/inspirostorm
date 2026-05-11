"""Web-URL sources: fetch a webpage, strip markup, ask GPT to keep only
the relevant body text, then persist it as a Scout *source* (same dict
shape as ``github_scan.fetch_repos`` / ``arxiv_scan.fetch_papers`` /
``uploads.save_upload``).

The persisted cleaned text lives under
``DATA_DIR/uploads/<source_id>.txt``; the lightweight source dict
returned to the frontend carries the cleaned text on ``preview_text`` so
the identity-generation prompt has rich context without re-reading the
file. ``knowledge.build_url_documents`` re-reads the on-disk file when
the user later attaches a knowledge base.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment
from openai import OpenAI

from . import storage

log = logging.getLogger("scout.web_scrape")

# Reuse the uploads dir so the cleaned text sits beside zip/pdf uploads
# and shares its retention / volume mount.
UPLOADS_DIR = storage.DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_FETCH_BYTES = int(5 * 1024 * 1024)          # 5 MB of HTML is plenty
MAX_RAW_CHARS_TO_LLM = 40_000                   # cap GPT input
PREVIEW_CHARS = 6000                            # truncated preview for identity prompt
FETCH_TIMEOUT = 20
PLAYWRIGHT_TIMEOUT_MS = 30_000                  # 30s ceiling for JS-rendering retry
# How big must the static-HTML <main>/<article> body be before we trust
# it? Smaller than this and we assume the page is JS-rendered (Astro,
# Next, etc.) and retry through Playwright. The Runway docs page that
# triggered this fallback had a <main> with 13 chars of static text.
SPA_MAIN_TEXT_MIN_CHARS = 200
USER_AGENT = "Scout/0.1 (webpage-to-avatar knowledge base)"

DEFAULT_CLEAN_MODEL = os.environ.get("WEB_SCRAPE_MODEL", "gpt-5.4")
FALLBACK_CLEAN_MODELS = ["gpt-5.2", "gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]

CLEAN_SYSTEM_PROMPT = """You are an extractor that turns a noisy webpage
dump into clean reading material.

You will be given the visible text of a webpage with HTML/CSS/JS already
removed but with lots of layout noise still present: navigation menus,
banners, cookie notices, footers, "subscribe to our newsletter" CTAs,
related-article rails, ad slots, breadcrumb trails, repeated site
chrome, social share buttons, comment counts, and so on.

Your job: return ONLY the main content body — the article, blog post,
documentation page, README, paper text, or product description that the
page is actually about. Preserve paragraph structure and headings as
plain text. Keep code blocks verbatim when present. Strip everything
that is site chrome, navigation, or promotional. If the page has a
clear title, put it on the very first line.

Hard rules:
- Output the cleaned text only. No preamble, no commentary, no
  markdown fences, no "Here is the cleaned text:".
- Do not summarise. Keep the original wording of the main content.
- If the page is genuinely empty or is just navigation (no article
  body), return the single line: NO_CONTENT
"""


@dataclass
class StoredUrl:
    source: dict
    cleaned_path: Path


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Reject obviously-unsafe URLs (loopback, link-local, private nets).

    This is a basic SSRF guard — not bulletproof against DNS rebinding,
    but it stops the easy mistakes (someone pasting http://localhost or
    http://169.254.169.254 into the form)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "could not parse URL"
    if parsed.scheme not in ("http", "https"):
        return False, "only http(s) URLs are supported"
    if not parsed.hostname:
        return False, "URL has no hostname"
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {e}"
    for fam, _stype, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False, f"refusing to fetch private/internal address ({ip_str})"
    return True, ""


def _fetch_html(url: str) -> tuple[str, str]:
    """Return ``(html_text, final_url)``. Raises ``ValueError`` on user
    error (bad URL / non-HTML response), ``RuntimeError`` on transport."""
    ok, reason = _is_safe_url(url)
    if not ok:
        raise ValueError(reason)
    try:
        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"},
            allow_redirects=True,
            stream=True,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"fetch failed: {e}") from e
    if resp.status_code >= 400:
        raise ValueError(f"server returned HTTP {resp.status_code}")
    ctype = (resp.headers.get("content-type") or "").lower()
    if "html" not in ctype and "text" not in ctype:
        raise ValueError(f"response is not HTML/text (content-type={ctype!r})")

    # Read up to MAX_FETCH_BYTES.
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_FETCH_BYTES:
            log.info("truncating fetch at %d bytes for %s", total, url)
            break
    body = b"".join(chunks)
    encoding = resp.encoding or "utf-8"
    try:
        html_text = body.decode(encoding, errors="replace")
    except LookupError:
        html_text = body.decode("utf-8", errors="replace")
    return html_text, resp.url


# ---------------------------------------------------------------------------
# Strip
# ---------------------------------------------------------------------------


_DROP_TAGS = ("script", "style", "noscript", "svg", "template", "iframe")


def _strip_to_text(html: str) -> tuple[str, str]:
    """Strip HTML/CSS/JS and return ``(plain_text, page_title)``."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    title_tag = soup.find("title")
    page_title = (title_tag.get_text(strip=True) if title_tag else "") or ""

    # Drop the <head> entirely — anything we needed (title) is captured.
    head = soup.find("head")
    if head:
        head.decompose()

    text = soup.get_text(separator="\n")
    text = _normalise_whitespace(text)
    return text, page_title


def _normalise_whitespace(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# SPA detection + Playwright fallback
# ---------------------------------------------------------------------------


def _looks_like_spa_shell(html: str) -> bool:
    """Heuristic: does this HTML look like an unhydrated SPA shell?

    True if there is a clear ``<main>`` or ``<article>`` element but it
    contains less than ``SPA_MAIN_TEXT_MIN_CHARS`` chars of text — that
    is the Astro/Next/React/etc. "shipped a skeleton, content arrives
    via JS" pattern.

    Pages without a main/article landmark are NOT flagged here — they
    may still be SPAs but we'd rather let the LLM-clean step decide
    (it returns NO_CONTENT for genuinely empty pages, which is the
    other trigger for the fallback).
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    landmark = soup.find("main") or soup.find("article")
    if not landmark:
        return False
    text_len = len(landmark.get_text(strip=True))
    return text_len < SPA_MAIN_TEXT_MIN_CHARS


def _fetch_html_via_playwright(url: str) -> tuple[str, str]:
    """Render ``url`` in headless Chromium and return ``(html, final_url)``.

    Raises ``RuntimeError`` with an install hint if Playwright or its
    browser binary aren't available. Any other rendering failure is
    re-raised as a ``RuntimeError`` for the caller to surface.
    """
    try:
        from playwright.sync_api import sync_playwright, Error as PWError
    except ImportError as e:
        raise RuntimeError(
            "playwright is not installed; install with: "
            "venv/bin/pip install playwright && venv/bin/playwright install chromium"
        ) from e

    log.info("retrying via Playwright (JS render): %s", url)
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except PWError as e:
                if "Executable doesn't exist" in str(e) or "browserType.launch" in str(e):
                    raise RuntimeError(
                        "chromium not downloaded; run: "
                        "venv/bin/playwright install chromium"
                    ) from e
                raise
            try:
                ctx = browser.new_context(user_agent=USER_AGENT)
                page = ctx.new_page()
                # networkidle waits until no more network activity for 500ms
                # — captures most SPAs after their initial data fetches.
                page.goto(url, wait_until="networkidle",
                          timeout=PLAYWRIGHT_TIMEOUT_MS)
                html = page.content()
                final_url = page.url
                return html, final_url
            finally:
                browser.close()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"playwright render failed: {e}") from e


# ---------------------------------------------------------------------------
# LLM clean
# ---------------------------------------------------------------------------


def _openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _llm_clean(raw_text: str) -> str:
    """Ask GPT-5.4 (with fallback chain) to keep only main-body content."""
    snippet = raw_text[:MAX_RAW_CHARS_TO_LLM]
    if len(raw_text) > MAX_RAW_CHARS_TO_LLM:
        log.info("truncating raw text from %d to %d chars before LLM clean",
                 len(raw_text), MAX_RAW_CHARS_TO_LLM)

    client = _openai_client()
    candidates = [DEFAULT_CLEAN_MODEL] + [
        m for m in FALLBACK_CLEAN_MODELS if m != DEFAULT_CLEAN_MODEL
    ]
    last_err: Optional[Exception] = None
    for model in candidates:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": snippet},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = RuntimeError(f"model {model} returned empty content")
                continue
            log.info("LLM-cleaned text via %s: %d → %d chars",
                     model, len(snippet), len(text))
            return text
        except Exception as e:
            last_err = e
            log.warning("clean via %s failed: %s", model, e)
            continue
    raise RuntimeError(f"all clean-model candidates failed; last: {last_err}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_and_clean(
    *,
    url: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> StoredUrl:
    """Fetch ``url``, strip + LLM-clean, persist, return a Scout source.

    Raises ``ValueError`` on user-visible errors (bad URL, non-HTML,
    empty result) and ``RuntimeError`` for transient transport / LLM
    failures.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("url is required")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    html, final_url = _fetch_html(url)
    used_js_render = False

    # SPA shell detection — if the static HTML has a near-empty <main>
    # element, we know the content is JS-rendered (Astro / Next / etc.)
    # and the bs4 strip will give us only nav + sidebar. Re-fetch via
    # Playwright before wasting an LLM call on garbage.
    if _looks_like_spa_shell(html):
        log.info("static HTML looks like an unhydrated SPA shell — retrying via Playwright")
        try:
            html, final_url = _fetch_html_via_playwright(url)
            used_js_render = True
        except RuntimeError as e:
            log.warning("Playwright retry failed, falling back to static HTML: %s", e)

    raw_text, page_title = _strip_to_text(html)
    if not raw_text.strip():
        raise ValueError("no readable text on this page")

    cleaned = _llm_clean(raw_text)

    # Fallback path: LLM said NO_CONTENT but we haven't tried JS rendering
    # yet. Some SPAs ship enough text that _looks_like_spa_shell doesn't
    # flag them (e.g. landmarks named differently) but still only show
    # nav. Retry with Playwright before giving up.
    if cleaned.strip().upper() == "NO_CONTENT" and not used_js_render:
        log.info("LLM reported NO_CONTENT — retrying via Playwright")
        try:
            html, final_url = _fetch_html_via_playwright(url)
            used_js_render = True
            raw_text, page_title = _strip_to_text(html)
            if raw_text.strip():
                cleaned = _llm_clean(raw_text)
        except RuntimeError as e:
            log.warning("Playwright retry failed: %s", e)

    if cleaned.strip().upper() == "NO_CONTENT":
        raise ValueError("page has no main content body to extract")

    source_id = f"url:{uuid.uuid4().hex}"
    cleaned_path = UPLOADS_DIR / f"{source_id.split(':', 1)[1]}.txt"
    cleaned_path.write_text(cleaned, encoding="utf-8")
    log.info("stored cleaned URL text url=%r chars=%d -> %s",
             final_url, len(cleaned), cleaned_path)

    effective_title = (title or "").strip() or page_title or final_url
    preview = cleaned
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS] + "\n\n…[truncated]"

    source = {
        "id": source_id,
        "source": "url",
        "title": effective_title,
        "subtitle": final_url,
        "description": (description or "").strip()
            or (cleaned[:240].replace("\n", " ").strip() + ("…" if len(cleaned) > 240 else "")),
        "url": final_url,
        "meta": {
            "page_title": page_title,
            "cleaned_path": str(cleaned_path),
            "raw_chars": len(raw_text),
            "cleaned_chars": len(cleaned),
            "js_rendered": used_js_render,
        },
        # Mirrors uploads.save_upload: identity-gen reads this via
        # main._readme_for, knowledge.build_url_documents reads the
        # on-disk file (which has the full cleaned text, not the
        # truncated preview).
        "preview_text": preview,
    }
    return StoredUrl(source=source, cleaned_path=cleaned_path)
