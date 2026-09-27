"""
paperswithcode.py

PWC's own API was retired by Meta in July 2025 — paperswithcode.com now
redirects to Hugging Face's trending papers page, and the old leaderboard
API is gone. This module fills the same slot (Source.PAPERSWITHCODE /
TrustTag.PWC_SOURCE) using Hugging Face's public, no-auth Papers API
instead: https://huggingface.co/api/papers/{arxiv_id}

Important distinction for the compiler: this does NOT give benchmark or
leaderboard numbers the way PWC used to. It confirms a paper is indexed
on HF and lists linked models/datasets/Spaces/GitHub repo — i.e. "does
an implementation exist," not "were the reported numbers reproduced."
The PWC_SOURCE tag rule should be worded to reflect that.

This is an ENRICHMENT step, not independent search: it takes arxiv_ids
already found by the arxiv agent and looks up artifact info for each.
HF's search endpoint exists but is capped (~20-25 results, no pagination)
and isn't a reliable enough primary retrieval path.
"""

import requests
from typing import List, Optional
from utils.models.classes import PooledChunk, Source

BASE_URL = "https://huggingface.co/api/papers"


def _chunk_from_response(arxiv_id: str, data: dict) -> Optional[PooledChunk]:
    if not data or "title" not in data:
        return None

    title = data.get("title", "")
    github = data.get("githubRepo")
    project_page = data.get("projectPage")
    models = data.get("models", []) or []
    datasets = data.get("datasets", []) or []
    spaces = data.get("spaces", []) or []

    if not (github or project_page or models or datasets or spaces):
        # Indexed on HF but nothing to actually report — not useful as a chunk.
        return None

    parts = []
    if github:
        parts.append(f"GitHub repository: {github}")
    if project_page:
        parts.append(f"Project page: {project_page}")
    if models:
        parts.append(f"{len(models)} linked model(s) on Hugging Face")
    if datasets:
        parts.append(f"{len(datasets)} linked dataset(s) on Hugging Face")
    if spaces:
        parts.append(f"{len(spaces)} linked demo Space(s) on Hugging Face")

    text = f"{title} — " + "; ".join(parts)

    return PooledChunk(
        source=Source.PAPERSWITHCODE,
        source_id=arxiv_id,
        title=title,
        section="linked_artifacts",
        score=0.0,  # filled later by the relevance gate, not by this fetcher
        text=text,
    )


def fetch_by_arxiv_id(arxiv_id: str, timeout: float = 5.0) -> Optional[PooledChunk]:
    """
    Look up implementation/artifact info for a single paper by arXiv ID.

    Returns None if the paper isn't indexed on HF, or is indexed but has
    no linked artifacts worth reporting — both are normal outcomes, not
    errors (HF Daily Papers only covers a subset, mostly recent + notable
    submissions within ~14 days of the arXiv post).
    """
    try:
        resp = requests.get(f"{BASE_URL}/{arxiv_id}", timeout=timeout)
    except requests.RequestException as e:
        print(f"⚠️  HF papers API error for {arxiv_id}: {e}")
        return None

    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        print(f"⚠️  HF papers API returned {resp.status_code} for {arxiv_id}")
        return None

    try:
        data = resp.json()
    except ValueError:
        return None

    return _chunk_from_response(arxiv_id, data)


def fetch_by_arxiv_ids(arxiv_ids: List[str]) -> List[PooledChunk]:
    """
    Batch wrapper — one request per ID (the endpoint has no batch mode).
    Meant to run AFTER the arxiv agent, enriching whichever papers it
    already found rather than searching independently.
    """
    chunks: List[PooledChunk] = []
    for arxiv_id in arxiv_ids:
        chunk = fetch_by_arxiv_id(arxiv_id)
        if chunk is not None:
            chunks.append(chunk)
    return chunks