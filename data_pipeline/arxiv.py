import arxiv
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from models.classes import Paper

FRESHNESS_DAYS: dict[str, Optional[int]] = {
    "high":   7,
    "medium": 30,
    "low":    None,
}


def _build_query(search_terms: List[str]) -> str:
    """
    Build a valid arXiv Lucene query. 
    Fetch papers from arXiv matching any of the search terms.
    Filters by date and deduplicates against seen_ids.

    THE CORE RULE — arXiv Lucene field scoping breaks on spaces:
      abs:flash attention  →  parses as  abs:flash AND attention
                           →  'attention' without field scope matches ALL attention papers
                           →  you get thousands of false positives

    So multi-word terms MUST use the all: field (full-text, no field parse ambiguity)
    or be hyphenated/camelCase single tokens.

    Strategy:
      - Single-token terms  → ti: (highest signal) + abs: (wider net)
      - Multi-word terms    → all: (safe fallback — full text search, no parse ambiguity)
      - Max 4 terms total to keep URL length reasonable
    """
    terms = list(dict.fromkeys(search_terms))[:4]

    if not terms:
        raise ValueError("search_terms is empty — analyzer returned no terms")

    single_token = [t for t in terms if " " not in t]
    multi_word   = [t for t in terms if " " in t]

    clauses = []

    # Single-token terms: field-scoped (precise)
    for t in single_token:
        clauses.append(f"ti:{t} OR abs:{t}")

    # Multi-word terms: all: field (safe — no Lucene parse ambiguity)
    for t in multi_word:
        clauses.append(f'all:"{t}"')  # quotes inside all: are safe, only abs:/ti: breaks

    return " OR ".join(f"({c})" for c in clauses)


def fetch_papers(
    search_terms: List[str],
    freshness: str = "medium",
    max_results: int = 20,
    seen_ids: set | None = None,
) -> List[Paper]:
    seen_ids = seen_ids or set()
    lookback_days = FRESHNESS_DAYS.get(freshness)
    cutoff: Optional[datetime] = None
    if lookback_days is not None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)

    query = _build_query(search_terms)

    arxiv_client = arxiv.Client(
        page_size=max_results,
        delay_seconds=1.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[Paper] = []
    try:
        for result in arxiv_client.results(search):
            pub = result.published
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)

            entry_id = result.entry_id

            if cutoff is not None and pub < cutoff:
                continue
            if entry_id in seen_ids:
                continue

            papers.append(Paper(
                title=result.title,
                authors=[a.name for a in result.authors],
                summary=result.summary,
                published=pub,
                pdf_url=str(result.pdf_url),
                arxiv_id=entry_id.split("/")[-1],
                link=entry_id,
            ))

    except arxiv.HTTPError as e:
        print(f"⚠️  arXiv HTTP error ({e.status}): {e}")
        print(f"   Query was: {query}")
        return []
    except Exception as e:
        print(f"⚠️  arXiv fetch error: {e}")
        return []

    return papers
