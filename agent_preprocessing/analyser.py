"""
Query analyzer— takes the raw user query and decides:
- What the user is asking about (the literal subject, not just generic terms)
- What the user's intent is (explain, compare, what's new, etc)
- What sources to check (arXiv, GitHub, news)
- How recent the results should be (freshness)
"""

import re
from typing import List
from models.classes import Source
from models.llm import chat_json

SYSTEM = "You are a research query analyzer. Always respond with valid JSON only. No preamble."

PROMPT_TEMPLATE = """Analyze this research query and return a JSON routing plan.

Query: {query}
Prior context: {context}

Return this exact JSON:
{{
  "mode": "known_paper|search|recall",
  "topic": "one-sentence description",
  "arxiv_ids": [],
  "search_terms": [],
  "sources": ["arxiv"],
  "freshness": "high|medium|low",
  "recall_mode": false
}}

━━━ MODE ━━━
"known_paper" → query names a specific paper/algorithm with a known arXiv ID. Never guess IDs.
"search" → broad topic or unknown ID
"recall" → "what did we find?", "what did we learn about X?"

━━━ SEARCH TERMS — CRITICAL ━━━
search_terms[0] MUST be the literal main subject of the query (the actual
named thing being asked about), cleaned of words like "papers"/"recent"/"explain".

  "Mamba architecture papers"     → terms[0] = "Mamba"
  "recent computer vision papers" → terms[0] = "computer vision"
  "how does FlashAttention work"  → terms[0] = "FlashAttention"

Add 2-3 supporting/related terms after it. NEVER drop the literal subject.
NEVER use noise words as terms: "paper", "research", "study", "method".

━━━ FRESHNESS ━━━
"low" → explain/what is/how does (no cutoff)  |  "medium" → recent (30d)  |  "high" → this week (7d)

━━━ SOURCES ━━━
"arxiv" → papers/theory  |  "github" → code  |  "web" → news

JSON only:"""


# Words that are NEVER part of the actual subject — small, stable, finite list
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "does", "do", "did", "how", "what", "why", "when", "explain", "tell",
    "me", "about", "recent", "recently", "latest", "new", "paper", "papers",
    "research", "work", "works", "architecture", "study", "studies",
}


def _extract_literal_subject(query: str) -> str:
    """
    Keep only non-stopword tokens, preserving original order and casing.
    Far more robust than filler-stripping — finite stopword set vs
    infinite sentence-structure edge cases.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query)
    kept = [t for t in tokens if t.lower() not in _STOPWORDS]
    return " ".join(kept) if kept else query.strip(" ?.!")


class QueryPlan:
    def __init__(self, data: dict, original_query: str = ""):
        self.mode: str               = data.get("mode", "search")
        self.topic: str              = data.get("topic", "")
        self.arxiv_ids: List[str]    = data.get("arxiv_ids", [])
        self.search_terms: List[str] = data.get("search_terms", [])
        self.sources: List[Source]   = [
            Source(s) for s in data.get("sources", ["arxiv"])
            if s in [src.value for src in Source]
        ]
        self.freshness: str    = data.get("freshness", "low")
        self.recall_mode: bool = self.mode == "recall"

        if self.mode == "search" and original_query:
            literal = _extract_literal_subject(original_query)
            terms_lower = [t.lower() for t in self.search_terms]
            # Only force-insert if no existing term already contains the subject
            if literal and not any(literal.lower() in t for t in terms_lower):
                self.search_terms.insert(0, literal)

        noise = {"paper", "papers", "research", "study", "method"}
        self.search_terms = [t for t in self.search_terms if t.lower() not in noise]

        seen, deduped = set(), []
        for t in self.search_terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        self.search_terms = deduped[:4]

    def __repr__(self):
        if self.mode == "known_paper":
            return f"QueryPlan(mode=known_paper, ids={self.arxiv_ids}, fallback={self.search_terms})"
        return f"QueryPlan(mode={self.mode}, terms={self.search_terms}, freshness={self.freshness})"


def analyze_query(query: str, context: str = "") -> QueryPlan:
    data = chat_json(
        PROMPT_TEMPLATE.format(query=query, context=context or "(none)"),
        system=SYSTEM,
    )
    return QueryPlan(data, original_query=query)