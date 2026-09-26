"""
Intent classifier — first gate in the pipeline.
 
PROJECT  → query references an ongoing codebase or prior research session
STANDALONE → self-contained general knowledge question

"""

from models.classes import QueryIntent
from models.llm import chat

SYSTEM_PROMPT = "You classify research queries. Respond with exactly one word." # we want a single word response to make it easy to parse
PROMPT_TEMPLATE = """Classify this query. Output exactly one word: PROJECT or STANDALONE.

PROJECT = explicitly references the user's OWN ongoing system by name.
Examples: "improve MY tokenizer", "my current model", "our training pipeline"

STANDALONE = general ML/CS knowledge question, even if the topic overlaps with the user's work.
Examples: "how does FlashAttention work?" → STANDALONE (asking about a paper/algorithm)
          "should I use FlashAttention in MY model?" → PROJECT (references their system)

Query: {query}

One word only:"""

def classify_intent(query: str) -> QueryIntent:
    raw = chat(PROMPT_TEMPLATE.format(query = query), system = SYSTEM_PROMPT, max_tokens=5)
    token = raw.strip().upper()
    if "PROJECT" in token:
        return QueryIntent.PROJECT
    return QueryIntent.STANDALONE