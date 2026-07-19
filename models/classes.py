from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from enum import Enum


class QueryIntent(str, Enum):
    PROJECT = "PROJECT"       # references ongoing codebase / prior sessions
    STANDALONE = "STANDALONE" # self-contained general knowledge (basically asking random questions)


class Source(str, Enum):
    ARXIV = "arxiv"
    GITHUB = "github"
    PAPERSWITHCODE = "paperswithcode"
    WEB = "web"


class TrustTag(str, Enum):
    """
    Maps each Source to the tag the compiler wraps its content in.
    Trust tier, not style: tells the compiler how much confidence
    it's allowed to state a claim with.
    """
    ARXIV_SOURCE = "ARXIV_SOURCE"      # peer-reviewed, section-anchored
    PWC_SOURCE = "PWC_SOURCE"          # structured benchmark/implementation data
    WEB_UNVERIFIED = "WEB_UNVERIFIED"  # blogs/docs/forums, no peer review


SOURCE_TRUST_TAG = {
    Source.ARXIV: TrustTag.ARXIV_SOURCE,
    Source.PAPERSWITHCODE: TrustTag.PWC_SOURCE,
    Source.GITHUB: TrustTag.PWC_SOURCE,
    Source.WEB: TrustTag.WEB_UNVERIFIED,
}


class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published: datetime
    pdf_url: str
    arxiv_id: str
    link: str
    relevance_score: float = 0.0   # filled by evidence ranker


class RouterDecision(BaseModel):
    """Output of classify_sources() — which agents to activate, and why."""
    needs_arxiv: bool = True
    needs_web: bool = False
    needs_pwc: bool = False
    sub_questions: List[str] = []
    reason: str = ""


class PooledChunk(BaseModel):
    """
    One retrieved unit from any source, normalized to a common shape
    so it can sit in one pooled context (and encode cleanly to TOON —
    uniform array of objects is TOON's sweet spot).
    """
    source: Source
    source_id: str      # arxiv_id, repo name, or domain
    title: str
    section: str = ""
    score: float = 0.0
    text: str

    @property
    def trust_tag(self) -> TrustTag:
        return SOURCE_TRUST_TAG[self.source]


class ResearchSession(BaseModel):
    query: str
    intent: QueryIntent
    search_terms: List[str]
    sources_used: List[Source]
    findings: str
    paper_ids: List[str]
    created_at: datetime = datetime.now()