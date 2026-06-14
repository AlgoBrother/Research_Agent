"""
arXiv source retriever.
Fixed version of your original code — same logic, clean interface.

What was broken in the original:
- `create_agent` import doesn't exist → crashed on import
- `self.client(messages)` syntax wrong for any real LLM SDK
- Redundant date check (checked cutoff twice)
- `_add_to_cache()` called but never defined
- Cache saved arxiv_id but papers_seen checked entry_id (different format)
"""

import arxiv
from datetime import datetime, timedelta
from typing import List

from models.classes import Paper

FRESHNESS_DAYS = {
    "high":   7,
    "medium": 30,
    "low":    365,
}


def fetch_papers(
    search_terms: List[str],
    freshness: str = "medium",
    max_results: int = 20,
    seen_ids: set | None = None,
) -> List[Paper]:
    """
    Fetch papers from arXiv matching any of the search terms.
    Filters by date and deduplicates against seen_ids.
    """
    seen_ids = seen_ids or set()
    lookback_days = FRESHNESS_DAYS.get(freshness, 30)
    cutoff = datetime.now(tz=datetime.now().astimezone().tzinfo) - timedelta(days=lookback_days)

    # "term1 OR term2 OR term3" — arXiv native syntax
    query = " OR ".join(f'"{t}"' for t in search_terms)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    arvix_client = arxiv.Client()

    papers: List[Paper] = []
    for result in arvix_client.results(search):
        # Normalize timezone for comparison
        pub = result.published
        if pub.tzinfo is None:
            from datetime import timezone
            pub = pub.replace(tzinfo=timezone.utc)

        entry_id = result.entry_id  # canonical ID e.g. "http://arxiv.org/abs/2401.12345v1"

        if pub < cutoff:
            continue
        if entry_id in seen_ids:
            continue

        papers.append(Paper(
            title=result.title,
            authors=[a.name for a in result.authors],
            summary=result.summary,
            published=pub,
            pdf_url=str(result.pdf_url),
            arxiv_id=result.entry_id.split("/")[-1],  # "2401.12345v1"
            link=entry_id,
        ))

    return papers