from models.classes import Source
from models.llm import chat_json
from typing import List

SYSTEM = "You are a research query analyzer. Always respond with valid JSON only. No preamble."

PROMPT_TEMPLATE = """Analyze this research query and output a JSON routing plan.

Query: {query}

Project context (may be empty):
{context}

Output this exact JSON structure:
{{
  "topic": "one-sentence description of what this is about",
  "search_terms": ["term1", "term2", "term3", "term4"],
  "sources": ["arxiv", "github", "web"],
  "freshness": "high|medium|low",
  "recall_mode": false
}}

Rules for sources (ordered by priority for this query):
- "arxiv"  → ML papers, algorithms, architectures, benchmarks, theory
- "github" → implementation, code, libraries, CLI tools, repos
- "web"    → news, blog posts, release announcements, how-to guides

Rules for freshness:
- "high"   → user needs info from the last 7 days (breaking news, latest releases)
- "medium" → last 30 days is fine
- "low"    → timeless: classic papers, algorithms, fundamentals

Set recall_mode=true ONLY if the query is asking about what was learned in a prior session
(e.g. "what did we find last time?", "what did we learn about X?").

Respond with JSON only:"""


class QueryPlan:
    def __init__(self, data: dict):
        self.topic: str = data.get("topic", "")
        self.search_terms: List[str] = data.get("search_terms", [])
        self.sources: List[Source] = [
            Source(s) for s in data.get("sources", ["arxiv"])
            if s in [src.value for src in Source]
        ]
        self.freshness: str = data.get("freshness", "medium")
        self.recall_mode: bool = data.get("recall_mode", False)

    def __repr__(self):
        return (
            f"QueryPlan(topic={self.topic!r}, "
            f"terms={self.search_terms}, "
            f"sources={[s.value for s in self.sources]}, "
            f"freshness={self.freshness})"
        )


def analyze_query(query: str, context: str = "") -> QueryPlan:
    data = chat_json(
        PROMPT_TEMPLATE.format(query=query, context=context or "(none)"),
        system=SYSTEM,
    )
    return QueryPlan(data)