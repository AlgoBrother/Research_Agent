"""
answer_generator.py — Synthesizes research answers using TOON context & targeted PDF excerpts.
"""

import re
from typing import List, Generator
from utils.models.classes import Paper
from utils.models.llm import chat, chat_stream
from utils.models.data_pipeline.pdf_fetcher import fetch_paper_text


TOON_SYSTEM = (
    "You are a careful, knowledgeable research assistant explaining ML and computer science concepts.\n"
    "You are provided with context formatted in TOON (Token-Oriented Object Notation).\n\n"
    "### TOON Reading Rules:\n"
    "1. Header layout: `context_chunks[N]{source,source_id,trust_tag,score,title,section,text}:`\n"
    "2. Each row contains comma-separated values mapping to those schema columns.\n\n"
    "### Output & Citation Requirements:\n"
    "- ONLY make claims supported by the provided TOON context.\n"
    "- Cite every factual claim, code reference, or mechanism using the exact `source_id` column value in inline brackets (e.g., `[2307.08691]` or `[github.com/Dao-AILab/flash-attention]`).\n"
    "- Include exact mathematical notation if present in the source text and explain it in plain language.\n\n"
    "### Trust Tiers — CRITICAL, read the trust_tag column and follow this exactly:\n"
    "- `ARXIV_SOURCE`: peer-reviewed papers. Theoretical truth. State claims directly.\n"
    "- `PWC_SOURCE`: confirms an implementation/model/dataset is indexed and linked — it does NOT mean "
    "reported benchmark numbers have been independently reproduced. State what's linked, don't imply verification beyond that.\n"
    "- `WEB_UNVERIFIED`: blogs, articles, forum posts — NOT peer-reviewed, NOT fact-checked. Any claim sourced "
    "ONLY from a WEB_UNVERIFIED chunk MUST be hedged explicitly — use phrasing like 'according to [source], "
    "though unverified' or 'publicly reported, but not independently confirmed'. NEVER state a WEB_UNVERIFIED "
    "claim with the same confidence as an ARXIV_SOURCE or PWC_SOURCE claim. This applies especially to "
    "specific numbers (scores, percentages, rankings, dates) — these are exactly where an unverified web "
    "source is most likely to be wrong or outdated, so hedge numbers from WEB_UNVERIFIED sources every time.\n"
    "- If the ONLY sources for a claim about a private/closed-source system are WEB_UNVERIFIED, and the "
    "numbers or details seem inconsistent with what you'd expect, say so plainly rather than repeating them "
    "as fact — e.g. 'multiple web sources report X, but this could not be verified against a primary source'.\n\n"
    "- Maintain a natural, direct, technically substantive tone without fluff or report headers."
)

TOON_ANSWER_PROMPT = """User's question: {query}

TOON CONTEXT:
{toon_context}

Write a direct, technically substantive answer with inline citations. Remember: hedge any claim
that comes ONLY from a WEB_UNVERIFIED source — do not state it with the same confidence as an
ARXIV_SOURCE or PWC_SOURCE claim.

Answer:"""

NO_EVIDENCE_TEMPLATE = """I couldn't find relevant papers or context for "{query}".
This could mean it's not indexed yet, needs more specific terms, or is too recent.
Share a link or arXiv ID and I'll fetch it directly."""


# =====================================================================
# PDF Targeted Extraction Helpers (Pre-TOON Context Enrichment)
# =====================================================================

def _chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """Split into paragraph-aware chunks of roughly chunk_size chars."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += "\n\n" + p
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _score_chunk(chunk: str, query_terms: List[str]) -> int:
    """Cheap keyword overlap score — counts term occurrences."""
    chunk_lower = chunk.lower()
    return sum(chunk_lower.count(t.lower()) for t in query_terms)


def extract_targeted_sections(
    pdf_url: str,
    query: str,
    query_terms: List[str] | None = None,
    max_pages: int = 10,
    top_chunks: int = 2,
) -> str:
    """
    Fetches the paper PDF, chunks it, and extracts key sections matching the user query.
    Used during context pooling before TOON compilation.
    """
    full_text = fetch_paper_text(pdf_url, max_pages=max_pages)
    if not full_text:
        return ""

    chunks = _chunk_text(full_text)
    if not chunks:
        return ""

    query_terms = query_terms or []
    all_terms = list(set(query_terms + re.findall(r"[A-Za-z]{4,}", query)))

    scored = sorted(
        ((c, _score_chunk(c, all_terms)) for c in chunks),
        key=lambda x: x[1],
        reverse=True,
    )

    selected = [c for c, score in scored[:top_chunks] if score > 0]
    if not selected:
        selected = chunks[:1]  # Fallback to introduction chunk

    return "\n\n[...]\n\n".join(selected)


# =====================================================================
# Main Answer Generation Entry Points
# =====================================================================

def generate_answer(
    query: str,
    toon_context: str,
    model: str = "qwen/qwen3.8-27b",
    max_tokens: int = 1500,
) -> str:
    """
    Primary generator: Accepts compiled TOON context string from Orchestrator
    and returns the full cited answer in one call. Kept for callers that
    don't need token-level streaming (e.g. concept fallback, recall summary).
    """
    if not toon_context or "context_chunks[0]" in toon_context:
        return NO_EVIDENCE_TEMPLATE.format(query=query)

    prompt = TOON_ANSWER_PROMPT.format(query=query, toon_context=toon_context)
    return chat(
        prompt,
        system=TOON_SYSTEM,
        model=model,
        max_tokens=max_tokens,
        temperature=0.2
    )


def generate_answer_stream(
    query: str,
    toon_context: str,
    model: str = "qwen/qwen3.8-27b",
    max_tokens: int = 1500,
) -> Generator[str, None, None]:
    """
    Streaming counterpart to generate_answer(). Yields text deltas as the
    model writes them, for callers (ResearchAgent.ask via on_token) that want
    to forward tokens to a live client instead of waiting for the full answer.

    No-context case still yields the full NO_EVIDENCE_TEMPLATE as a single
    chunk rather than a real stream, since there's nothing to generate.
    """
    if not toon_context or "context_chunks[0]" in toon_context:
        yield NO_EVIDENCE_TEMPLATE.format(query=query)
        return

    prompt = TOON_ANSWER_PROMPT.format(query=query, toon_context=toon_context)
    yield from chat_stream(
        prompt,
        system=TOON_SYSTEM,
        model=model,
        max_tokens=max_tokens,
        temperature=0.2
    )


def generate_concept_answer(query: str, model: str = "qwen/qwen3.8-27b") -> str:
    """Fallback generator for general knowledge/concept questions without context."""
    prompt = f"Explain this concept clearly and accurately: {query}\n\nGeneral knowledge, not a specific paper. Note uncertainty if relevant."
    return chat(prompt, system=TOON_SYSTEM, model=model, max_tokens=800, temperature=0.3)