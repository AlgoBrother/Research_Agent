from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from utils.timeline import timed_stage
from data_pipeline.router.router import QueryRouter, RouterDecision
from data_pipeline.mappers import map_to_pooled_chunks
from data_pipeline.toon_compiler import ToonCompiler


class Orchestrator:
    def __init__(self, router: QueryRouter, arxiv_client, pwc_client, web_client):
        self.router = router
        self.arxiv = arxiv_client
        self.pwc = pwc_client
        self.web = web_client
        self.compiler = ToonCompiler() 

    def run(self, query: str) -> Dict[str, Any]:
        latency = {}
        raw_results = {}

        # 1. Intent Classification / Routing
        with timed_stage("router", latency):
            decision: RouterDecision = self.router.route(query)

        # 2. Parallel Fetching (arXiv and Web are independent HTTP calls)
        with ThreadPoolExecutor() as executor:
            futures = {}

            if decision.needs_arxiv:
                futures["arxiv"] = executor.submit(self._fetch_arxiv, query, latency)
            
            if decision.needs_web:
                futures["web"] = executor.submit(self._fetch_web, query, latency)

            # Wait for parallel executions to complete
            for tier, future in futures.items():
                raw_results[tier] = future.result()

        # 3. Sequential Fetching (PWC depends on arXiv results)
        if decision.needs_pwc and raw_results.get("arxiv"):
            arxiv_docs = raw_results["arxiv"]
            arxiv_ids = []
            for doc in arxiv_docs:
                aid = getattr(doc, "arxiv_id", doc.get("arxiv_id") if isinstance(doc, dict) else getattr(doc, "id", None))
                if aid:
                    arxiv_ids.append(aid)

            if arxiv_ids:

                with timed_stage("pwc", latency):
                    raw_results["pwc"] = self.pwc.fetch(arxiv_ids)

        # 4. Normalization & TOON Compilation (FIX 3: Runs for ALL queries)
        with timed_stage("map_pooled_chunks", latency):
            pooled_chunks = map_to_pooled_chunks(raw_results)

        with timed_stage("toon_compile", latency):
            toon_context = self.compiler.compile(pooled_chunks)

        # FIX 4: Return compiled TOON context alongside raw results
        return {
            "query": query,
            "decision": decision,
            "data": raw_results,
            "toon_context": toon_context,
            "latency": latency
        }

    def _fetch_arxiv(self, query: str, latency: dict) -> List[Any]:
        with timed_stage("arxiv", latency):
            return self.arxiv.search(query)

    def _fetch_web(self, query: str, latency: dict) -> List[Any]:
        with timed_stage("web", latency):
            return self.web.search(query)