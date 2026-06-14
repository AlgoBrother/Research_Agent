"""
Memory module.
Two things live here:
  1. PaperCache    — file-based set of seen arxiv IDs
  2. SessionMemory — ChromaDB store for full research sessions (enables recall queries)
"""

import json
from pathlib import Path
from datetime import datetime
from models.classes import Paper

from models.classes import ResearchSession


# ─────────────────────────────────────────────
# 1. Paper dedup cache
# ─────────────────────────────────────────────


 
CACHE_FILE = Path("papers_seen.json")
 
 
class PaperCache:
    def __init__(self):
        self._data: dict[str, dict] = self._load()
 
    def _load(self) -> dict:
        if CACHE_FILE.exists():
            try:
                return json.loads(CACHE_FILE.read_text())
            except json.JSONDecodeError:
                print("⚠️  papers_seen.json corrupted — starting fresh")
                return {}
        return {}
 
    def _save(self):
        CACHE_FILE.write_text(json.dumps(self._data, indent=2))
 
    def seen(self, paper_id: str) -> bool:
        return paper_id in self._data
 
    def mark_seen(self, paper_id: str, title: str = ""):
        self._data[paper_id] = {
            "title": title,
            "added": datetime.now().isoformat(),
        }
        self._save()
 
    def mark_seen_bulk(self, paper_ids: list[str], papers=None):
        """
        paper_ids: list of entry_id strings
        papers:    optional list of Paper objects (to store titles)
        """
        title_map = {}
        if papers:
            title_map = {p.link: p.title for p in papers}
 
        for pid in paper_ids:
            if pid not in self._data:
                self._data[pid] = {
                    "title": title_map.get(pid, ""),
                    "added": datetime.now().isoformat(),
                }
        self._save()
 
    @property
    def ids(self) -> set:
        return set(self._data.keys())
 


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