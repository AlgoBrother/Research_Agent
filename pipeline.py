"""
pipeline.py:  main end-to-end pipeline orchestration. Takes a user query, runs it through the (testing purpose only
"""

import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

from models.classes import QueryIntent, Source, ResearchSession
from agent_preprocessing.intent_classifier import classify_intent
from agent_preprocessing.analyser          import analyze_query, QueryPlan
from agent_preprocessing.ranker            import rank_papers
from agent_preprocessing.report_generator  import generate_report, save_report
from data_pipeline.arxiv                   import fetch_papers, fetch_by_ids
from data_pipeline.database                import PaperCache, SessionMemory


# Words that signal "this query references the previous turn"
FOLLOWUP_SIGNALS = re.compile(
    r"\b(that|it|this|those|these|compare|versus|vs\.?|"
    r"similar|related|also|further|more about|expand|elaborate)\b",
    re.IGNORECASE,
)


def _is_followup(query: str) -> bool:
    return bool(FOLLOWUP_SIGNALS.search(query))


class Research_Agent:
    def __init__(self, top_k: int = 6):
        self.top_k           = top_k
        self.cache           = PaperCache()
        self.session_memory  = SessionMemory()
        self.history: list[dict] = []
        self._session_seen: set  = set()

    def run(self, query: str, verbose: bool = True) -> str:
        self.history.append({"role": "user", "content": query})
        report = self._run(query=query, verbose=verbose)
        self.history.append({"role": "agent", "content": report[:300]})
        return report

    def reset_history(self):
        self.history = []
        self._session_seen = set()
        print("🔄 Conversation reset.")

    def _run(self, query: str, verbose: bool = True) -> str:
        def log(msg: str):
            if verbose: print(msg)

        log("\n🔍 Classifying intent...")
        intent = classify_intent(query)
        log(f"   → {intent.value}")

        # ── KEY FIX: only build history context when it's actually relevant ──
        context = self._build_context(intent, query, log)

        log("🧠 Analyzing query...")
        plan: QueryPlan = analyze_query(query=query, context=context)
        log(f"   mode     : {plan.mode}")
        log(f"   topic    : {plan.topic}")
        if plan.mode == "known_paper":
            log(f"   IDs      : {plan.arxiv_ids}")
            log(f"   fallback : {plan.search_terms}")
        else:
            log(f"   terms    : {plan.search_terms}")
            log(f"   freshness: {plan.freshness}")

        if plan.recall_mode:
            return self._handle_recall(query, log)

        papers = self._retrieve(plan, log)
        if not papers:
            return (
                "No papers found for this query.\n"
                "Suggestions: try broader terms, or ask about a specific named paper."
            )

        rank_terms = plan.search_terms if plan.search_terms else [plan.topic]
        if plan.mode == "known_paper" and len(papers) <= self.top_k:
            top_papers = papers
        else:
            top_papers = rank_papers(papers, rank_terms, top_k=self.top_k)
        log(f"   → {len(top_papers)} papers selected")

        log("✍️  Generating report...")
        report = generate_report(query=query, topic=plan.topic, papers=top_papers)

        self._save(query, intent, plan, top_papers, report)
        save_report(query=query, report=report, papers=top_papers)
        log("✅ Done.\n")

        return report

    def _build_context(self, intent: QueryIntent, query: str, log) -> str:
        """
        Only inject history when:
          - intent is PROJECT (references ongoing work), OR
          - query has explicit follow-up signal words
        Otherwise return empty string — treat as a fresh, unrelated query.
        """
        parts = []

        should_use_history = (
            intent == QueryIntent.PROJECT or _is_followup(query)
        )

        if should_use_history and len(self.history) > 1:
            recent = self.history[-4:-1]
            if recent:
                turns = "\n".join(f"{t['role'].upper()}: {t['content'][:200]}" for t in recent)
                parts.append(f"=== Recent conversation ===\n{turns}")
                log("   (using conversation history — follow-up detected)")
        elif len(self.history) > 1:
            log("   (new topic — ignoring conversation history)")

        if intent == QueryIntent.PROJECT:
            log("📂 Retrieving project context...")
            sessions = self.session_memory.recall(query, n=2)
            ctx = self.session_memory.format_for_context(sessions)
            if ctx:
                log(f"   Found {len(sessions)} prior session(s)")
                parts.append(ctx)
            else:
                log("   No prior sessions found")

        return "\n\n".join(parts)

    def _retrieve(self, plan: QueryPlan, log) -> list:
        if plan.mode == "known_paper" and plan.arxiv_ids:
            log(f"📌 Fetching by ID: {plan.arxiv_ids}")
            papers = fetch_by_ids(plan.arxiv_ids)
            log(f"   Got {len(papers)} paper(s)")
            if not papers and plan.search_terms:
                log("   ID fetch empty — falling back to search...")
                papers = fetch_papers(
                    search_terms=plan.search_terms,
                    freshness=plan.freshness,
                    max_results=15,
                    seen_ids=self._session_seen,
                )
                log(f"   Fallback: {len(papers)} papers")
            return papers

        all_papers = []
        for source in plan.sources:
            if source == Source.ARXIV:
                log(f"📡 Searching arXiv...")
                papers = fetch_papers(
                    search_terms=plan.search_terms,
                    freshness=plan.freshness,
                    max_results=20,
                    seen_ids=self._session_seen,
                )
                log(f"   Got {len(papers)} papers")
                all_papers.extend(papers)
            elif source == Source.GITHUB:
                log("⚠️  GitHub not yet implemented")
            elif source == Source.WEB:
                log("⚠️  Web not yet implemented")
        return all_papers

    def _handle_recall(self, query: str, log) -> str:
        log("💾 Searching session memory...")
        sessions = self.session_memory.recall(query, n=5)
        if sessions:
            return self.session_memory.format_for_context(sessions)
        return "No prior research found on this topic."

    def _save(self, query, intent, plan, papers, report):
        self._session_seen.update(p.link for p in papers)
        self.cache.mark_seen_bulk(paper_ids=[p.link for p in papers], papers=papers)
        session = ResearchSession(
            query=query,
            intent=intent,
            search_terms=plan.search_terms,
            sources_used=plan.sources,
            findings=report,
            paper_ids=[p.arxiv_id for p in papers],
        )
        self.session_memory.save(session)


if __name__ == "__main__":
    agent = Research_Agent(top_k=6)
    print("🔬 Research Agent ready. Commands: 'quit', 'reset'\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "reset":
            agent.reset_history()
            continue

        report = agent.run(query)
        print(f"\n{'─'*60}")
        print(report)
        print(f"{'─'*60}\n")