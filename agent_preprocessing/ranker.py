"""
Evidence ranker.
Scores papers BEFORE sending them to the LLM.
Prevents dumping 20 semi-relevant abstracts into a 8k context window.
 
Scoring is intentionally heuristic — no extra LLM call needed here.
Three signals:
  1. Relevance  — keyword overlap with search terms
  2. Freshness  — recency decay (half-life = 90 days)
  3. (future)   — citation count when available PLANNED
"""

import math
from datetime import datetime, timezone
from typing import List
from models.classes import Paper

def _relevance_score(paper: Paper, search_items: List[str]) -> float:
    text_title = paper.title.lower()
    text_body = paper.summary.lower()
    terms = [term.lower() for term in search_items]

    title_hits = sum(1 for term in terms if term in text_title) # how many search terms appear in the title
    body_hits = sum(1 for term in terms if term in text_body) # how many search terms appear in the body

    if not terms:
        return 0.5 # neutral score if no search terms provided
    
    return min(1.0, (title_hits * 2 + body_hits) / (len(terms) * 3)) # weighted score: title hits count double

def _freshness_score(paper: Paper) -> float:
    """Exponential decay. Papers older than ~6 months score near 0."""
    pub = paper.published
    if pub.tzinfo is None: # tzinfo is a naive datetime, assume UTC
        pub = pub.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    days_old = max(0, (now - pub).days)
    half_life = 90  # days
    return math.exp(-0.693 * days_old / half_life) # -0.693 is ln(2) for half-life decay

def rank_papers(
        papers: List[Paper], 
        search_items: List[str], 
        top_k : int = 8, 
        relevance_weight: float = 0.6,
        freshness_weight: float = 0.4
        
    ) -> List[Paper]:
    for paper in papers:
        rel_score = _relevance_score(paper, search_items)
        fresh_score = _freshness_score(paper)
        paper.relevance_score = (rel_score * relevance_weight) + (fresh_score * freshness_weight) # combine scores with weights

    ranked = sorted(papers, key = lambda p: p.relevance_score, reverse=True) # sort papers by combined score
    return ranked[:top_k] # return the top_k papers

    
