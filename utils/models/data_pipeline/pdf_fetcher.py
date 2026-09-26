"""
pdf_fetcher.py

Downloads the PDF and extracts real text content, instead of only
ever using the abstract.
"""

import httpx
import io
from pypdf import PdfReader


def fetch_paper_text(pdf_url: str, max_pages: int = 6) -> str:
    """
    Download a PDF and extract text from the first max_pages pages.

    Why limit pages: the intro + method sections (where the actual
    explanation of HOW something works lives) are almost always in
    the first 4-6 pages. Related work, experiments, and references
    take up the back half and add little value for "explain X" queries
    while burning context window tokens.

    Returns extracted text, or empty string on failure (caller should
    fall back to abstract-only).
    """
    try:
        resp = httpx.get(pdf_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️  PDF download failed: {e}")
        return ""

    try:
        reader = PdfReader(io.BytesIO(resp.content))
        pages_to_read = min(max_pages, len(reader.pages))
        text_parts = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"⚠️  PDF parse failed: {e}")
        return ""


def fetch_paper_text_safe(pdf_url: str, max_pages: int = 6, max_chars: int = 12000) -> str:
    """
    Wrapper with character cap --- protects against extremely long
    extracted text blowing past the LLM's context window.
    12000 chars ≈ 3000 tokens, leaves room for prompt + multiple papers.
    """
    text = fetch_paper_text(pdf_url, max_pages=max_pages)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated...]"
    return text