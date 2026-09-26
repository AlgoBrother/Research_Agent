import os
import requests
from typing import List
from utils.models.classes import PooledChunk, Source

TAVILY_URL = "https://api.tavily.com/search"


def _domain_from_url(url: str) -> str:
    stripped = url.split("://")[-1]
    return stripped.split("/")[0]


def _result_to_chunk(result: dict) -> PooledChunk:
    url = result.get("url", "")
    return PooledChunk(
        source=Source.WEB,
        source_id=_domain_from_url(url),
        title=result.get("title", ""),
        section=url,
        score=result.get("score", 0.0),
        text=result.get("content", ""),
    )


def fetch_web_results(query: str, max_results: int = 5, timeout: float = 10.0) -> List[PooledChunk]:
    """
    Search the web via Tavily, return normalized PooledChunks tagged
    Source.WEB.

    Fails soft everywhere: missing API key, network errors, non-200
    responses, or bad JSON all return [] rather than raising, so a
    web-search outage never breaks the arxiv-only path.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("⚠️  TAVILY_API_KEY not set — skipping web search")
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",  # not gonna use advanced since it costs more and is slower
    }

    try:
        resp = requests.post(TAVILY_URL, json=payload, timeout=timeout)
    except requests.RequestException as e:
        print(f"⚠️  Tavily request error: {e}")
        return []

    if resp.status_code != 200:
        print(f"⚠️  Tavily returned {resp.status_code}: {resp.text[:200]}")
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    results = data.get("results", []) or []
    return [_result_to_chunk(r) for r in results if r.get("content")]