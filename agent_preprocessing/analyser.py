"""
Query analyzer

Generate terms the way arXiv actually tokenizes them.
"""

from typing import List
from models.classes import Source
from models.llm import chat_json

SYSTEM = "You are a research query analyzer. Always respond with valid JSON only. No preamble."

PROMPT_TEMPLATE = """Analyze this research query and return a JSON routing plan.

Query: {query}
Project context: {context}

Return this exact JSON:
{{
  "topic": "one-sentence description",
  "search_terms": ["term1", "term2", "term3", "term4"],
  "sources": ["arxiv"],
  "freshness": "high|medium|low",
  "recall_mode": false
}}

--- SEARCH TERMS (CRITICAL RULES) ---
arXiv uses Lucene. Field-scoped terms (ti:, abs:) BREAK on spaces:
  abs:flash attention  →  Lucene sees: abs:flash AND attention
  →  "attention" without scope matches EVERY attention paper
  →  result: 10,000 irrelevant papers

RULE: search_terms must be ONE of:
  a) A single CamelCase token:   "FlashAttention", "GQA", "RoPE", "SwiGLU"
  b) A hyphenated token:         "IO-aware", "context-length", "multi-head"
  c) A short 2-word phrase (OK in all: field): "flash attention", "ring attention"

Generate 3-4 terms. Examples by query type:

"how does FlashAttention work?"
→ ["FlashAttention", "IO-aware", "tiling", "flash attention"]

"explain speculative decoding"
→ ["speculative decoding", "draft model", "token speculation", "parallel decoding"]

"improve tokenizer efficiency"
→ ["BPE", "tokenizer", "subword", "vocabulary"]

"what is GQA?"
→ ["GQA", "grouped-query attention", "multi-query attention", "KV cache"]

--- SOURCES ---
"arxiv"  → ML papers, algorithms, architecture, theory
"github" → code, repos, implementations
"web"    → news, releases, blogs

--- FRESHNESS ---
"low"    → "how does X work?", "explain X", "what is X?", classic algorithms
"medium" → "recent work on X", "latest papers on X"
"high"   → "today", "this week", "latest release"

Default to "low" for any explanatory or architectural question.

--- RECALL MODE ---
recall_mode=true ONLY for: "what did we find last time?", "what did we learn about X?"

JSON only:"""


class QueryPlan:
    def __init__(self, data: dict):
        self.topic: str             = data.get("topic", "")
        self.search_terms: List[str] = data.get("search_terms", [])
        self.sources: List[Source]  = [
            Source(s) for s in data.get("sources", ["arxiv"])
            if s in [src.value for src in Source]
        ]
        self.freshness: str   = data.get("freshness", "low")
        self.recall_mode: bool = data.get("recall_mode", False)

    def __repr__(self):
        return (
            f"QueryPlan(topic={self.topic!r}, "
            f"terms={self.search_terms}, "
            f"freshness={self.freshness})"
        )


def analyze_query(query: str, context: str = "") -> QueryPlan:
    data = chat_json(
        PROMPT_TEMPLATE.format(query=query, context=context or "(none)"),
        system=SYSTEM,
    )
    return QueryPlan(data)