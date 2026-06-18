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
from models.classes import Paper


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


def papers_are_relevant(query: str, search_terms: List[str], papers: List[Paper],
                         min_overlap_ratio: float = 0.15) -> bool:
    """
    Check whether retrieved papers actually relate to the query.

    Heuristic: build a vocabulary from the query + search_terms, then
    check what fraction of THAT vocabulary appears across the papers'
    titles+abstracts combined. If overlap is too low, the retrieval
    likely matched on noise (e.g. arXiv returning broad/unrelated
    results for a vague or non-research query).

    Returns True if papers pass the relevance bar, False otherwise.
    """
    if not papers:
        return False

    query_vocab = _tokenize(query) | _tokenize(" ".join(search_terms))
    query_vocab -= _STOP_FOR_RELEVANCE
    query_vocab = {w for w in query_vocab if len(w) > 2}  # drop tiny tokens

    if not query_vocab:
        return False

    paper_text = " ".join(f"{p.title} {p.summary}" for p in papers)
    paper_vocab = _tokenize(paper_text)

    overlap = query_vocab & paper_vocab
    ratio = len(overlap) / len(query_vocab)

    return ratio >= min_overlap_ratio