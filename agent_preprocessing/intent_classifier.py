"""
Intent classifier — first gate in the pipeline.
 
PROJECT  → query references an ongoing codebase or prior research session
STANDALONE → self-contained general knowledge question

"""

from models.classes import QueryIntent
from models.llm import chat

SYSTEM_PROMPT = "You classify research queries. Respond with exactly one word." # we want a single word response to make it easy to parse
PROMPT_TEMPLATE = """Classify this query. Output exactly one word: PROJECT or STANDALONE.
 
PROJECT = references an ongoing codebase, project state, architecture, or prior session.
Examples:
  - "how do I improve Maya's attention mechanism?"
  - "what did we learn about tokenizer fertility last time?"
  - "should I use GQA or MHA in my current model?"
 
STANDALONE = self-contained general knowledge, no project dependency.
Examples:
  - "how does FlashAttention work?"
  - "what is symbolic algebra?"
  - "explain the difference between GQA and MHA"
 
Query: {query}
 
One word only:"""

def classify_intent(query: str) -> QueryIntent:
    raw = chat(PROMPT_TEMPLATE.format(query = query), system = SYSTEM_PROMPT, max_tokens=5)
    token = raw.strip().upper()
    if "PROJECT" in token:
        return QueryIntent.PROJECT
    return QueryIntent.STANDALONE