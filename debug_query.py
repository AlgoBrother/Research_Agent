"""
Query debugger — run this when you get bad results.
Shows exactly what the agent is doing at each step without making LLM calls for report gen.

Usage:
  python debug_query.py "how does FlashAttention reduce memory usage?"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from agent_preprocessing.intent_classifier import classify_intent
from agent_preprocessing.analyser import analyze_query
from data_pipeline.arxiv import _build_query, FRESHNESS_DAYS
from datetime import datetime, timedelta, timezone


def debug(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    # Step 1: Intent
    print("\n[1] INTENT CLASSIFICATION")
    intent = classify_intent(query)
    print(f"    → {intent.value}")

    # Step 2: Query plan
    print("\n[2] QUERY ANALYSIS")
    plan = analyze_query(query)
    print(f"    topic      : {plan.topic}")
    print(f"    terms      : {plan.search_terms}")
    print(f"    sources    : {[s.value for s in plan.sources]}")
    print(f"    freshness  : {plan.freshness}")
    print(f"    recall_mode: {plan.recall_mode}")

    # Step 3: Show the actual arXiv query string
    print("\n[3] ARXIV QUERY CONSTRUCTION")
    arxiv_query = _build_query(plan.search_terms)
    print(f"    query string: {arxiv_query!r}")
    
    lookback = FRESHNESS_DAYS.get(plan.freshness)
    if lookback is None:
        print(f"    date filter : NONE (freshness=low — searches all time)")
    else:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback)
        print(f"    date filter : papers after {cutoff.strftime('%Y-%m-%d')} ({lookback}d)")

    print(f"\n    arXiv URL preview:")
    import urllib.parse
    encoded = urllib.parse.quote(arxiv_query)
    print(f"    https://export.arxiv.org/api/query?search_query={encoded}&max_results=10")
    print(f"\n    ↑ paste this URL in your browser to verify results before running the full agent")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "how does FlashAttention reduce memory usage?"
    debug(q)