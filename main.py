import sys
import os
sys.path.insert(0, os.path.dirname(__file__)) # Add the current directory to the Python path
from models.classes import QueryIntent, Source, ResearchSession
from models.llm import chat

from agent_preprocessing.intent_classifier import classify_intent
from agent_preprocessing.analyser   import analyze_query, QueryPlan
from agent_preprocessing.ranker     import rank_papers
from agent_preprocessing.report_generator   import generate_report, save_report

from data_pipeline.arxiv import fetch_papers
from data_pipeline.database import PaperCache, SessionMemory

class ResearchAgent:
    def __init__(self, top_k : int = 8):
        self.top_k = top_k
        self.cache = PaperCache()
        self.session_memory = SessionMemory()

    def run(self, query: str, verbose: bool = False) -> str:
        def log(msg:str):
            if verbose:
                print(msg)

        log("\nClassifying intent...\n")
        intent = classify_intent(query)
        log(f"Intent classified as: {intent}\n")

        # Context Retrieval for Project Queries
        context = ""
        if intent == QueryIntent.PROJECT:
            log("📂 Retrieving project context...")
            prior_sessions = self.session_memory.recall(query, n=3) # get relevant prior sessions
            context = self.session_memory.format_for_context(prior_sessions)
            if context:
                log(f"   Found {len(prior_sessions)} prior session(s)")
            else:
                log("   No prior sessions found — continuing without context")

        # Query Analysis
        log("\nAnalyzing query and generating search terms...\n")
        plan: QueryPlan = analyze_query(query=query, context=context)
        log(f"Topic: {plan.topic}\n")
        log(f"Search terms: {plan.search_terms}\n")
        log(f"Sources: {[s.value for s in plan.sources]}\n")

        # Recalling (prior session search )
        if plan.recall_mode:
            log("\nRecalling relevant prior sessions...\n")
            sessions = self.session_memory.recall(query, n=3)
            if sessions:
                log(f"   Found {len(sessions)} relevant prior session(s)")
                finding = self.session_memory.format_for_context(sessions)
                print(f"\nRelevant prior findings:\n{finding}\n")
                return finding
            else:
                log("\nNo relevant prior sessions found.")
                return "No relevant prior sessions found."
            
        # Source retrieval 
        all_papers = []
        for source in plan.sources:
            if source == Source.ARXIV:
                log(f"Fetching from arXiv...")
                papers = fetch_papers(
                    search_terms=plan.search_terms,
                    freshness=plan.freshness,
                    max_results=20,
                    seen_ids=self.cache.ids,
                )
                log(f"   Got {len(papers)} new papers")
                all_papers.extend(papers)
            elif source == Source.GITHUB:
                log("⚠️  GitHub source not yet implemented — skipping")
            elif source == Source.WEB:
                log("⚠️  Web source not yet implemented — skipping")
        if not all_papers:
            log("\nNo papers found from any source.")
            return "No papers found from any source."
            
        # Evidence ranking
        log("\nRanking papers by relevance...\n")
        top_ranked_papers = rank_papers(all_papers, plan.search_terms, top_k=self.top_k)
        log(f"Chosen top {len(top_ranked_papers)} papers:\n")

        # Generate report
        report = generate_report(query=query, topic=plan.topic, papers=top_ranked_papers)

        session = ResearchSession(
            query=query,
            intent=intent,
            search_terms=plan.search_terms,
            sources_used=plan.sources,
            findings=report,
            paper_ids=[p.arxiv_id for p in top_ranked_papers],
        )
        self.session_memory.save(session)
        self.cache.mark_seen_bulk(
            paper_ids=[p.link for p in top_ranked_papers],
            papers=top_ranked_papers
        )

        # Save report to disk
        save_report(query=query, report=report, papers=top_ranked_papers)
        log("\nReport generated and saved.\n")
        log("\n" + "="*50 + "\n")
        return report

if __name__ == "__main__":
    agent = ResearchAgent(top_k=6)
    report = agent.run(
        "Explain Weight Sharing Regularization",
        verbose=True   # ← see what's happening
    )
    print(report)      # ← see the actual output

