"""
answer_generator.py — adds targeted section extraction.

Generating Answers in the chat pipeline
"""

import re
from typing import List
from models.classes import Paper
from models.llm import chat
from data_pipeline.pdf_fetcher import fetch_paper_text

SYSTEM = (
    "You are a careful, knowledgeable research assistant explaining ML "
    "concepts using actual paper content. You ONLY make claims you can "
    "support with the provided text. Use real technical detail — "
    "equations, named techniques, specific mechanisms — not vague "
    "restatements. Include math/formulas if the source text has them, "
    "explained in plain language alongside."
)

ANSWER_PROMPT = """User's question: {query}

RELEVANT PAPER EXCERPTS (targeted to your question, not the whole paper):
{evidence}

SUPPORTING ABSTRACTS (other related papers, less detail):
{abstract_evidence}

Write a direct, technically substantive answer.

Rules:
- Use the excerpts for real mechanistic/technical detail.
- Include math notation if present in the source — explain it in plain words too.
- Cite papers inline: [Title](url)
- If the excerpts don't actually address the question, say so plainly.
- Natural tone, no padding, no report-style headers.

Answer:"""

NO_EVIDENCE_TEMPLATE = """I couldn't find a paper on arXiv matching "{query}".
This could mean it's not on arXiv, needs more specific terms, or is too recent.
Share a link or arXiv ID and I'll fetch it directly."""


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


def _extract_targeted_sections(
    pdf_url: str,
    query: str,
    query_terms: List[str],
    max_pages: int = 10,
    top_chunks: int = 3,
) -> str:
    """
    Fetch the paper, chunk it, and keep only the chunks most relevant
    to the user's specific question — not the whole intro by default.
    """
    full_text = fetch_paper_text(pdf_url, max_pages=max_pages)
    if not full_text:
        return ""

    chunks = _chunk_text(full_text)
    if not chunks:
        return ""

    # Always include query words themselves as scoring terms too
    all_terms = list(set(query_terms + re.findall(r"[A-Za-z]{4,}", query)))

    scored = sorted(
        ((c, _score_chunk(c, all_terms)) for c in chunks),
        key=lambda x: x[1],
        reverse=True,
    )

    # If nothing scored above 0, fall back to the first 2 chunks (intro)
    selected = [c for c, score in scored[:top_chunks] if score > 0]
    if not selected:
        selected = chunks[:2]

    return "\n\n[...]\n\n".join(selected)


def _format_abstracts(papers: List[Paper]) -> str:
    if not papers:
        return "(none)"
    return "\n".join(
        f"- {p.title} ({', '.join(p.authors[:2])}, {p.published.strftime('%Y-%m')}): {p.summary[:180]}..."
        for p in papers
    )


def generate_answer(
    query: str,
    papers: List[Paper],
    search_terms: List[str] | None = None,
    model: str = "llama-3.3-70b-versatile",
    full_text_count: int = 2,
) -> str:
    if not papers:
        return NO_EVIDENCE_TEMPLATE.format(query=query)

    query_terms = search_terms or []
    primary = papers[:full_text_count]
    rest    = papers[full_text_count:]

    evidence_blocks = []
    for p in primary:
        targeted = _extract_targeted_sections(p.pdf_url, query, query_terms)
        if targeted:
            evidence_blocks.append(
                f"### {p.title}\n({', '.join(p.authors[:2])}, {p.published.strftime('%Y-%m')})\n"
                f"URL: {p.pdf_url}\n\n{targeted}"
            )
        else:
            evidence_blocks.append(
                f"### {p.title} (full text unavailable, abstract only)\n"
                f"URL: {p.pdf_url}\n\n{p.summary}"
            )

    evidence = "\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
    abstract_evidence = _format_abstracts(rest)

    prompt = ANSWER_PROMPT.format(query=query, evidence=evidence, abstract_evidence=abstract_evidence)
    return chat(prompt, system=SYSTEM, model=model, max_tokens=1500, temperature=0.3)


def generate_concept_answer(query: str, model: str = "llama-3.3-70b-versatile") -> str:
    prompt = f"Explain this concept clearly and accurately: {query}\n\nGeneral knowledge, not a specific paper. Note uncertainty if relevant."
    return chat(prompt, system=SYSTEM, model=model, max_tokens=800, temperature=0.3)