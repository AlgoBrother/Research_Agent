"""
relevance_gate.py

Two gates that sit between retrieval and answer generation:

1. is_research_query() — cheap heuristic, catches greetings/chitchat
   BEFORE wasting an LLM call on analysis at all.

2. papers_are_relevant() — after retrieval, checks if the returned
   papers actually share meaningful vocabulary with the query.
   If not, treat it as "no results" rather than feeding noise to
   the answer generator.
"""

import re
from typing import List
from models.classes import Paper, PooledChunk, Source


# Common greetings / chitchat that should never trigger research pipeline
_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|good morning|good afternoon|good evening|wassup|greetings|howdy|what's new|what's going on|"
    r"how are you|what's up|thanks|thank you|ok|okay|cool|nice|great|"
    r"bye|goodbye|see ya)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_research_query(query: str) -> bool:
    """
    Returns False for greetings/chitchat/empty input that shouldn't
    trigger the retrieval pipeline at all.
    """
    q = query.strip()
    if len(q) < 3:
        return False
    if _CHITCHAT_PATTERNS.match(q):
        return False
    return True


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# Words too generic to count as a "relevance match" on their own
_STOP_FOR_RELEVANCE = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
    "are", "this", "that", "paper", "papers", "model", "models", "with",
    "using", "based", "approach", "method", "via",
}


# Web content is noisier than arXiv/PWC, so it needs a higher overlap
# bar before we trust it enters the pooled context at all.
SOURCE_RELEVANCE_THRESHOLD = {
    Source.ARXIV: 0.15,
    Source.PAPERSWITHCODE: 0.12,
    Source.GITHUB: 0.12,
    Source.WEB: 0.20,
}


def _extract_text(item) -> str:
    """Works across Paper (.summary) and PooledChunk (.text)."""
    title = getattr(item, "title", "")
    body = getattr(item, "summary", None)
    if body is None:
        body = getattr(item, "text", "")
    return f"{title} {body}"


def items_are_relevant(query: str, search_terms: List[str], items: List,
                        min_overlap_ratio: float = 0.15) -> bool:
    """
    Generalized relevance check — works over Paper OR PooledChunk objects,
    so the same gate can run per-source (arxiv/pwc/web) before pooling,
    not just once at the end over arxiv results.

    Heuristic: build a vocabulary from the query + search_terms, then
    check what fraction of THAT vocabulary appears across the items'
    title+body combined. If overlap is too low, the retrieval likely
    matched on noise. Returns True if items pass the relevance bar.
    """
    if not items:
        return False

    query_vocab = _tokenize(query) | _tokenize(" ".join(search_terms))
    query_vocab -= _STOP_FOR_RELEVANCE
    query_vocab = {w for w in query_vocab if len(w) > 2}  # drop tiny tokens

    if not query_vocab:
        return False

    combined_text = " ".join(_extract_text(item) for item in items)
    item_vocab = _tokenize(combined_text)

    overlap = query_vocab & item_vocab
    ratio = len(overlap) / len(query_vocab)

    return ratio >= min_overlap_ratio


def source_is_relevant(query: str, search_terms: List[str], chunks: List[PooledChunk]) -> bool:
    """
    Same check as items_are_relevant(), but picks the threshold based on
    the chunks' source (they should all share one source when called
    per-agent). Falls back to 0.15 if the list is empty or mixed.
    """
    if not chunks:
        return False
    threshold = SOURCE_RELEVANCE_THRESHOLD.get(chunks[0].source, 0.15)
    return items_are_relevant(query, search_terms, chunks, min_overlap_ratio=threshold)


def papers_are_relevant(query: str, search_terms: List[str], papers: List[Paper],
                         min_overlap_ratio: float = 0.15) -> bool:
    """Kept for backward compatibility — existing call sites still work unchanged."""
    return items_are_relevant(query, search_terms, papers, min_overlap_ratio)