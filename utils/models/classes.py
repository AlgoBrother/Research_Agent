from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from enum import Enum
from pydantic import Field


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
    needs_arxiv: bool = Field(description="True for research papers, ML architectures, academic concepts.")
    needs_pwc: bool = Field(description="True if official code, benchmarks, or paper implementations are needed.")
    needs_web: bool = Field(description="True for recent news, general web context, or non-academic queries.")
    sub_questions: List[str] = Field(
        default_factory=list, 
        description="1-3 targeted sub-questions if the query is multi-hop or complex."
    )
    reasoning: str = Field(description="Brief explanation of why these tiers were selected.")


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