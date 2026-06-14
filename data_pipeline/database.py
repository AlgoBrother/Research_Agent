"""
Memory module.
Two things live here:
  1. PaperCache    — file-based set of seen arxiv IDs (your original logic, cleaned up)
  2. SessionMemory — ChromaDB store for full research sessions (enables recall queries)

ChromaDB runs fully locally, no API key needed. pip install chromadb.
"""

import json
from pathlib import Path
from datetime import datetime

from models.classes import ResearchSession


# ─────────────────────────────────────────────
# 1. Paper dedup cache (flat file, fast)
# ─────────────────────────────────────────────

CACHE_FILE = Path("papers_seen.txt")


class PaperCache:
    def __init__(self):
        self._seen: set[str] = self._load()

    def _load(self) -> set[str]:
        if CACHE_FILE.exists():
            return set(CACHE_FILE.read_text().splitlines())
        return set()

    def _save(self):
        CACHE_FILE.write_text("\n".join(self._seen))

    def seen(self, paper_id: str) -> bool:
        return paper_id in self._seen

    def mark_seen(self, paper_id: str):
        self._seen.add(paper_id)
        self._save()

    def mark_seen_bulk(self, paper_ids: list[str]):
        self._seen.update(paper_ids)
        self._save()

    @property
    def ids(self) -> set[str]:
        return self._seen


# ─────────────────────────────────────────────
# 2. Session memory (ChromaDB, semantic search)
# ─────────────────────────────────────────────

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class SessionMemory:
    """
    Stores and retrieves full research sessions.
    Falls back to a JSON file if ChromaDB isn't installed yet.
    """

    def __init__(self, persist_dir: str = ".chroma"):
        self._use_chroma = _CHROMA_AVAILABLE
        if self._use_chroma:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._col = self._client.get_or_create_collection("research_sessions")
        else:
            self._fallback_file = Path("sessions_fallback.jsonl")
            print("⚠️  ChromaDB not installed — using JSON fallback. Run: pip install chromadb")

    def save(self, session: ResearchSession):
        doc = json.dumps({
            "query":        session.query,
            "intent":       session.intent.value,
            "findings":     session.findings,
            "search_terms": session.search_terms,
            "sources":      [s.value for s in session.sources_used],
            "paper_ids":    session.paper_ids,
            "created_at":   session.created_at.isoformat(),
        })

        if self._use_chroma:
            self._col.add(
                documents=[doc],
                ids=[f"session_{datetime.now().timestamp()}"],
                metadatas=[{"query": session.query, "type": "research_session"}],
            )
        else:
            with self._fallback_file.open("a") as f:
                f.write(doc + "\n")

    def recall(self, query: str, n: int = 3) -> list[dict]:
        """Return the n most relevant prior sessions for this query."""
        if self._use_chroma:
            count = self._col.count()
            if count == 0:
                return []
            results = self._col.query(
                query_texts=[query],
                n_results=min(n, self._col.count()),
                where={"type": "research_session"},
            )
            docs = results.get("documents", [[]])[0]
            return [json.loads(d) for d in docs]
        else:
            # Fallback: return last n sessions (no semantic search)
            if not self._fallback_file.exists():
                return []
            lines = self._fallback_file.read_text().strip().splitlines()
            return [json.loads(l) for l in lines[-n:]]

    def format_for_context(self, sessions: list[dict]) -> str:
        """Format recalled sessions as a context string for the LLM."""
        if not sessions:
            return ""
        parts = ["=== Prior research sessions ==="]
        for s in sessions:
            parts.append(
                f"Query: {s['query']}\n"
                f"Findings: {s['findings'][:400]}...\n"
                f"Terms used: {', '.join(s.get('search_terms', []))}"
            )
        return "\n\n".join(parts)