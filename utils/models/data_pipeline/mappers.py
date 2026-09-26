# mappers.py
from typing import Dict, Any, List
from models.classes import PooledChunk, Source, Paper

def map_to_pooled_chunks(raw_results: Dict[str, Any]) -> List[PooledChunk]:
    """
    Normalizes heterogeneous fetcher outputs (arXiv Paper objects, 
    PWC dicts, Web dicts) into a unified List[PooledChunk].
    """
    chunks: List[PooledChunk] = []

    # 1. Process arXiv Papers
    for paper in raw_results.get("arxiv", []):
        if isinstance(paper, Paper):
            aid = paper.arxiv_id
            title = paper.title
            text = paper.summary
            score = paper.relevance_score or 0.90
        elif isinstance(paper, dict):
            aid = paper.get("arxiv_id", "arxiv_unknown")
            title = paper.get("title", "")
            text = paper.get("summary", "")
            score = paper.get("relevance_score", 0.90)
        else:
            continue

        chunks.append(PooledChunk(
            source=Source.ARXIV,
            source_id=aid,
            title=title,
            section="Abstract",
            score=score,
            text=text
        ))

    # 2. Process Papers With Code (PWC)
    for repo in raw_results.get("pwc", []):
        repo_id = repo.get("repo_url") or repo.get("arxiv_id") or "pwc_repo"
        title = f"Implementation for {repo.get('arxiv_id', 'paper')}"
        stars = repo.get("stars", 0)
        framework = repo.get("framework", "PyTorch")
        desc = repo.get("description", "")
        
        text_payload = f"Framework: {framework} | Stars: {stars} | Repo: {repo_id} | Notes: {desc}"

        chunks.append(PooledChunk(
            source=Source.PAPERSWITHCODE,
            source_id=repo_id,
            title=title,
            section="Code Repository",
            score=0.88,
            text=text_payload
        ))

    # 3. Process Web Results
    for site in raw_results.get("web", []):
        url = site.get("url", "web_unknown")
        chunks.append(PooledChunk(
            source=Source.WEB,
            source_id=url,
            title=site.get("title", "Web Source"),
            section="",
            score=site.get("score", 0.75),
            text=site.get("snippet", "")
        ))

    return chunks