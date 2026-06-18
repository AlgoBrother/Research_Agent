import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

from models.classes import QueryIntent, Source, ResearchSession
from agent_preprocessing.intent_classifier import classify_intent
from agent_preprocessing.analyser          import analyze_query, QueryPlan
from agent_preprocessing.ranker            import rank_papers
from agent_preprocessing.answer_generator  import generate_answer, generate_concept_answer
from agent_preprocessing.relevance_gate    import is_research_query, papers_are_relevant
from data_pipeline.arxiv                   import fetch_papers, fetch_by_ids
from data_pipeline.database                import PaperCache, SessionMemory
from models.llm                            import chat


FOLLOWUP_SIGNALS = re.compile(
    r"\b(that|it|this|those|these|compare|versus|vs\.?|"
    r"similar|related|also|further|more about|expand|elaborate)\b",
    re.IGNORECASE,
)
AMBIGUOUS_REFERENT = re.compile(
    r"^\s*(tell me (more )?about (this|that|it)|"
    r"explain (this|that|it)$|"
    r"(this|that) paper|"
    r"more (about|on) (this|that|it))\b",
    re.IGNORECASE,
)


def _is_followup(query: str) -> bool:
    return bool(FOLLOWUP_SIGNALS.search(query))


def _is_garbage_query(query: str) -> bool:
    q = query.strip()
    if re.match(r'^https?://', q):
        return True
    if len(q) < 4:
        return True
    alpha_ratio = sum(c.isalpha() for c in q) / max(len(q), 1)
    return alpha_ratio < 0.3


def _is_ambiguous_followup(query: str, last_papers: list) -> bool:
    if not AMBIGUOUS_REFERENT.match(query.strip()):
        return False
    return len(last_papers) != 1


class ResearchAgent:
    def __init__(self, top_k: int = 6):
        self.top_k          = top_k
        self.cache          = PaperCache()
        self.session_memory = SessionMemory()
        self.history: list[dict] = []
        self._session_seen: set  = set()
        self._last_papers: list  = []

    def ask(self, query: str, on_step=None) -> dict:
        def step(msg):
            if on_step: on_step(msg)

        if _is_garbage_query(query):
            return {"answer": "That doesn't look like a research question. Try asking about a paper, technique, or topic.", "papers": [], "plan": None}

        if not is_research_query(query):
            self.history.append({"role": "user", "content": query})
            answer = "Hey! Ask me about a paper, ML technique, or research topic."
            self.history.append({"role": "agent", "content": answer})
            return {"answer": answer, "papers": [], "plan": None}

        if _is_ambiguous_followup(query, self._last_papers):
            if len(self._last_papers) > 1:
                titles = "\n".join(f"- {p.title}" for p in self._last_papers[:5])
                answer = f"I discussed a few papers last turn — which one did you mean?\n\n{titles}"
            else:
                answer = "Which paper or topic are you referring to?"
            return {"answer": answer, "papers": [], "plan": None}

        self.history.append({"role": "user", "content": query})

        step("Classifying intent...")
        intent = classify_intent(query)
        context = self._build_context(intent, query, step)

        step("Analyzing query...")
        plan: QueryPlan = analyze_query(query=query, context=context)
        step(f"   mode={plan.mode} terms={plan.search_terms} freshness={plan.freshness}")

        if plan.recall_mode:
            step("Searching memory...")
            answer = self._handle_recall_readable(query, step)
            self.history.append({"role": "agent", "content": answer[:300]})
            self._last_papers = []
            return {"answer": answer, "papers": [], "plan": plan}

        step("Searching arXiv (adaptive)...")
        papers = self._retrieve(plan, step)
        step(f"   retrieved {len(papers)} paper(s)")

        if plan.mode == "search" and papers:
            if not papers_are_relevant(query, plan.search_terms, papers):
                step("   not relevant — discarding")
                papers = []

        if not papers:
            answer = self._answer_without_papers(query, plan)
            self.history.append({"role": "agent", "content": answer[:300]})
            self._last_papers = []
            return {"answer": answer, "papers": [], "plan": plan}

        rank_terms = plan.search_terms if plan.search_terms else [plan.topic]
        top_papers = papers if (plan.mode == "known_paper" and len(papers) <= self.top_k) \
            else rank_papers(papers, rank_terms, top_k=self.top_k)

        step("Reading papers + writing answer...")
        answer = generate_answer(query, top_papers, search_terms=plan.search_terms)

        self._save(query, intent, plan, top_papers, answer)
        self.history.append({"role": "agent", "content": answer[:300]})
        self._last_papers = top_papers

        return {"answer": answer, "papers": top_papers, "plan": plan}

    def _answer_without_papers(self, query: str, plan: QueryPlan) -> str:
        if plan.mode == "known_paper":
            concept = generate_concept_answer(query)
            return f"Couldn't pull the exact paper right now. Here's what I know:\n\n{concept}\n\n*(general knowledge, not from a retrieved paper)*"
        return f"Couldn't find arXiv papers matching \"{query}\" even after widening the search. Try rephrasing, or share a link/arXiv ID."

    def run(self, query: str, verbose: bool = True) -> str:
        return self.ask(query, on_step=(print if verbose else None))["answer"]

    def reset_history(self):
        self.history = []
        self._session_seen = set()
        self._last_papers = []

    def _build_context(self, intent, query: str, log) -> str:
        parts = []
        if (intent == QueryIntent.PROJECT or _is_followup(query)) and len(self.history) > 1:
            recent = self.history[-4:-1]
            if recent:
                turns = "\n".join(f"{t['role'].upper()}: {t['content'][:200]}" for t in recent)
                parts.append(f"=== Recent conversation ===\n{turns}")
        if intent == QueryIntent.PROJECT:
            sessions = self.session_memory.recall(query, n=2)
            ctx = self.session_memory.format_for_context(sessions)
            if ctx:
                parts.append(ctx)
        return "\n\n".join(parts)

    def _retrieve(self, plan: QueryPlan, log) -> list:
        if plan.mode == "known_paper" and plan.arxiv_ids:
            papers = fetch_by_ids(plan.arxiv_ids)
            if not papers and plan.search_terms:
                papers = fetch_papers(plan.search_terms, plan.freshness, 15, self._session_seen)
            return papers
        all_papers = []
        for source in plan.sources:
            if source == Source.ARXIV:
                all_papers.extend(fetch_papers(plan.search_terms, plan.freshness, 20, self._session_seen))
        return all_papers

    def _handle_recall_readable(self, query: str, step) -> str:
        sessions = self.session_memory.recall(query, n=5)
        if not sessions:
            return "I don't have any prior research on this topic yet."
        raw = self.session_memory.format_for_context(sessions)
        prompt = f'The user asked: "{query}"\n\nPrior notes:\n{raw}\n\nSummarize in 3-5 natural sentences. No headers.'
        return chat(prompt, system="You summarize prior research conversationally.", max_tokens=400)

    def _save(self, query, intent, plan, papers, answer):
        self._session_seen.update(p.link for p in papers)
        self.cache.mark_seen_bulk(paper_ids=[p.link for p in papers], papers=papers)
        session = ResearchSession(
            query=query, intent=intent, search_terms=plan.search_terms,
            sources_used=plan.sources, findings=answer,
            paper_ids=[p.arxiv_id for p in papers],
        )
        self.session_memory.save(session)