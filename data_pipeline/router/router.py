from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from models.classes import RouterDecision

class QueryRouter:
    def __init__(self, client: OpenAI, model: str = "llama-3.1-8b-instant"):
        self.client = client
        self.model = model

    def route(self, query: str) -> RouterDecision:
        system_prompt = (
            "You are a routing agent for a research assistant.\n"
            "Analyze the user query and decide which data tiers to activate (arxiv, paperswithcode, web).\n\n"
            "Rules:\n"
            "1. Set needs_arxiv=True if the query asks about papers, ML architectures, or academic concepts.\n"
            "2. Set needs_pwc=True if the query asks for code, implementations, benchmarks, or GitHub repos.\n"
            "3. Set needs_web=True if the query asks for recent tech news, documentation, or general non-academic info.\n"
            "4. If the query is multi-hop or complex, split it into 1-3 targeted sub_questions tailored for the selected sources."
        )

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.0,            # Force deterministic decisions
                max_completion_tokens=200,  # Minimize routing latency
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                response_format=RouterDecision  
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Routing error: {str(e)}. Falling back to default routing.")
            # Safe Fallback: Activate arXiv and Web if routing fails
            return RouterDecision(
                needs_arxiv=True,
                needs_web=True,
                needs_pwc=False,
                sub_questions=[query],
                reasoning=f"Fallback activated due to routing error: {str(e)}"
            )