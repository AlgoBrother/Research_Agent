from typing import List
import json
from openai import OpenAI
from models.classes import RouterDecision


SYSTEM_PROMPT = (
    "You are a routing agent for a research assistant.\n"
    "Analyze the user query and decide which data tiers to activate (arxiv, paperswithcode, web).\n\n"
    "Rules:\n"
    "1. Set needs_arxiv=True if the query asks about papers, ML architectures, or academic concepts.\n"
    "2. Set needs_pwc=True if the query asks for code, implementations, benchmarks, or GitHub repos.\n"
    "3. Set needs_web=True if the query asks for recent tech news, documentation, or general non-academic info.\n"
    "4. IMPORTANT: Set needs_web=True whenever the query is about a CLOSED-SOURCE model with "
    "no public paper — e.g. Claude, GPT-4, Gemini, GLM (if discussing a proprietary variant). "
    "For these, arxiv alone cannot answer the question; web sources are required even for "
    "architecture/technical questions, since there is no paper to cite.\n"
    "5. If the query is multi-hop or complex, split it into 1-3 targeted sub_questions tailored for the selected sources.\n\n"
    "Example: \"How does Claude's architecture differ from GLM?\" — Claude has no public paper, "
    "so this needs BOTH arxiv (for GLM, if it has one) AND web (for Claude, and to catch any "
    "GLM variant that also lacks a formal paper). needs_arxiv=True, needs_web=True."
)

# Fallback-mode prompt : we apply the same rules, but spells out the exact JSON shape
# since there's no schema enforcement backing this path.
FALLBACK_PROMPT_SUFFIX = """

Respond with ONLY a JSON object, no markdown, no prose, in exactly this shape:
{"needs_arxiv": true, "needs_pwc": false, "needs_web": false, "sub_questions": [], "reasoning": "..."}"""


class QueryRouter:
    def __init__(self, client: OpenAI, model: str = "llama-3.1-8b-instant"):
        self.client = client
        self.model = model

    def route(self, query: str) -> RouterDecision:
        try:
            return self._route_structured(query)
        except Exception as e:
            print(f"Structured routing failed ({str(e)[:120]}...) — falling back to json_object mode.")
            try:
                return self._route_json_object(query)
            except Exception as e2:
                print(f"json_object routing ALSO failed: {e2}. Using static fallback.")
                return RouterDecision(
                    needs_arxiv=True, needs_web=True, needs_pwc=False,
                    sub_questions=[query],
                    reasoning=f"Both routing attempts failed: {str(e2)[:150]}",
                )

    def _route_structured(self, query: str) -> RouterDecision:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.0,
            max_completion_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format=RouterDecision,
        )
        return completion.choices[0].message.parsed

    def _route_json_object(self, query: str) -> RouterDecision:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_completion_tokens=250,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + FALLBACK_PROMPT_SUFFIX},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"}, 
        )
        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)  
        return RouterDecision(**data)  