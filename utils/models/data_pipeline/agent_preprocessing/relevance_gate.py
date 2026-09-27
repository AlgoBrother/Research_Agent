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
from utils.models.classes import Paper, PooledChunk, Source


# Common greetings / chitchat that should never trigger research pipeline
_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|good morning|good afternoon|good evening|"
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


def _norm_for_anchor(s: str) -> str:
    """Strips everything but alphanumerics so hyphen/space variance
    (SWE-bench vs SWE bench vs SWEbench) doesn't cause false negatives."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _anchor_present(anchor_term: str, combined_text: str) -> bool:
    """
    Checks whether the query's designated literal subject (search_terms[0],
    which analyze_query() already guarantees is "the actual named thing
    being asked about") genuinely appears in the retrieved text.

    This exists because word-overlap alone lets off-topic papers through
    when they share generic ML vocabulary with the query (e.g. a context-
    pruning paper passing a "SWE-bench" query because both mention
    "model"/"coding"/"benchmark") — the anchor is never actually present,
    but the overlap ratio still clears the bar.

    Only enforced for SHORT anchors (<=3 words). A longer anchor is
    almost always a sign term extraction produced a broken clause rather
    than a real subject (e.g. "Claude s differ from" instead of "Claude")
    — requiring that exact multi-word fragment to appear verbatim would
    reject genuinely relevant results, since it will never literally
    appear anywhere. In that case, skip this gate and rely on the
    overlap-ratio check alone, same as before this gate existed.
    """
    if not anchor_term:
        return True
    if len(anchor_term.split()) > 3:
        return True  # anchor too noisy to trust — don't block on it
    if len(anchor_term) <= 3:
        return True  # too short/generic to check reliably either
    return _norm_for_anchor(anchor_term) in _norm_for_anchor(combined_text)


def items_are_relevant(query: str, search_terms: List[str], items: List,
                        min_overlap_ratio: float = 0.15) -> bool:
    """
    Generalized relevance check — works over Paper OR PooledChunk objects,
    so the same gate can run per-source (arxiv/pwc/web) before pooling,
    not just once at the end over arxiv results.

    Two gates, both must pass:
    1. Vocabulary overlap — the existing heuristic.
    2. Anchor presence — the query's literal subject (search_terms[0])
       must actually appear in the text, not just generic shared
       vocabulary. This catches the recurring failure where an
       off-topic paper passes #1 on words like "model"/"benchmark"
       alone (see: SWE-Pruner Pro passing a SWE-bench query).
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

    if ratio < min_overlap_ratio:
        return False

    anchor = search_terms[0] if search_terms else ""
    return _anchor_present(anchor, combined_text)


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