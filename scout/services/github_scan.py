"""Scan a GitHub user's public repos."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

GITHUB_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_repos(
    username: str,
    *,
    language: Optional[str] = None,
    days: Optional[int] = None,
    limit: int = 30,
) -> list[dict]:
    """Return up to ``limit`` of the user's public repos.

    Filters:
      - ``language``: case-insensitive match against the repo's primary language.
      - ``days``: only repos updated in the last N days (uses ``updated_at``).
    """
    if not username:
        raise ValueError("username is required")

    repos: list[dict] = []
    page = 1
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
        if days
        else None
    )

    while len(repos) < limit and page < 10:
        url = f"{GITHUB_API}/users/{username}/repos"
        params = {"per_page": 100, "page": page, "sort": "updated", "direction": "desc"}
        resp = requests.get(url, headers=_headers(), params=params, timeout=20)
        if resp.status_code == 404:
            raise LookupError(f"GitHub user '{username}' not found")
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        for r in batch:
            if r.get("fork"):
                continue
            if language and (r.get("language") or "").lower() != language.lower():
                continue
            if cutoff:
                updated = datetime.fromisoformat(
                    r["updated_at"].replace("Z", "+00:00")
                )
                if updated < cutoff:
                    # Sorted desc by updated_at, so we can short-circuit
                    return _project(repos[:limit])
            repos.append(r)
            if len(repos) >= limit:
                break
        page += 1

    return _project(repos[:limit])


def fetch_readme(full_name: str, *, max_chars: int = 6000) -> str:
    """Fetch the README for ``owner/repo`` as plain text.

    Returns an empty string if the repo has no README or the request fails
    (we never want missing README to break a scan). The result is truncated
    to ``max_chars`` so we don't blow out LLM context on monorepos.
    """
    if not full_name or "/" not in full_name:
        return ""
    url = f"{GITHUB_API}/repos/{full_name}/readme"
    headers = _headers()
    headers["Accept"] = "application/vnd.github.raw"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException:
        return ""
    if resp.status_code != 200:
        return ""
    text = resp.text or ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…[truncated]"
    return text


def _project(repos: list[dict]) -> list[dict]:
    """Keep only the fields the frontend needs."""
    out: list[dict] = []
    for r in repos:
        out.append(
            {
                "id": f"gh:{r['full_name']}",
                "source": "github",
                "title": r["name"],
                "subtitle": r.get("full_name"),
                "description": r.get("description") or "",
                "url": r.get("html_url"),
                "meta": {
                    "language": r.get("language"),
                    "stars": r.get("stargazers_count", 0),
                    "forks": r.get("forks_count", 0),
                    "updated_at": r.get("updated_at"),
                    "topics": r.get("topics", []),
                },
            }
        )
    return out
