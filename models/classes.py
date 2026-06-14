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
    WEB = "web"


class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published: datetime
    pdf_url: str
    arxiv_id: str
    link: str
    relevance_score: float = 0.0   # filled by evidence ranker


class ResearchSession(BaseModel):
    query: str
    intent: QueryIntent
    search_terms: List[str]
    sources_used: List[Source]
    findings: str
    paper_ids: List[str]
    created_at: datetime = datetime.now()