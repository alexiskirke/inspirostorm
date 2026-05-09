"""Scan arXiv for recent papers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import arxiv


def fetch_papers(
    query: str,
    *,
    categories: Optional[list[str]] = None,
    days: Optional[int] = None,
    limit: int = 30,
) -> list[dict]:
    """Search arXiv and return up to ``limit`` recent papers.

    ``query`` is matched against title/abstract.
    ``categories`` filters by arXiv category (e.g. ``cs.AI``).
    ``days`` keeps only papers published in the last N days.
    """
    parts: list[str] = []
    if query.strip():
        parts.append(f"all:{query.strip()}")
    if categories:
        cat_clause = " OR ".join(f"cat:{c}" for c in categories)
        parts.append(f"({cat_clause})")
    if not parts:
        raise ValueError("Provide a query or at least one category")

    search = arxiv.Search(
        query=" AND ".join(parts),
        max_results=max(limit * 3, 30),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
        if days
        else None
    )

    out: list[dict] = []
    client = arxiv.Client(page_size=100, delay_seconds=1.0, num_retries=2)
    for r in client.results(search):
        if cutoff and r.published < cutoff:
            # Results are descending by date so we can stop scanning
            break
        out.append(
            {
                "id": f"arxiv:{r.entry_id.rsplit('/', 1)[-1]}",
                "source": "arxiv",
                "title": r.title.strip(),
                "subtitle": ", ".join(a.name for a in r.authors[:4])
                + (" et al." if len(r.authors) > 4 else ""),
                "description": (r.summary or "").strip(),
                "url": r.entry_id,
                "meta": {
                    "categories": list(r.categories),
                    "published": r.published.isoformat(),
                    "primary_category": r.primary_category,
                },
            }
        )
        if len(out) >= limit:
            break
    return out
