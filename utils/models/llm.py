import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None  # Lazy initialization

REASONING_MODELS = {
    "qwen/qwen3.6-27b", "qwen/qwen3-32b", "qwen-qwq-32b",
    "openai/gpt-oss-120b", "openai/gpt-oss-20b",
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>", re.IGNORECASE)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
    return _client


def get_router(client: OpenAI | None = None):
    """
    Instantiates and returns QueryRouter using the shared client.
    Import kept local to avoid a circular import (data_pipeline.router.router
    may itself import from models.llm) — only pay that cost if this is
    actually called.
    """
    from data_pipeline.router.router import QueryRouter
    if client is None:
        client = get_client()
    return QueryRouter(client=client, model="llama-3.1-8b-instant")


def chat(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    model: str = "llama-3.1-8b-instant",  # non-reasoning default
    temperature: float = 0.2,
    max_tokens: int = 1024
) -> str:
    client = get_client()
    kwargs = dict(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )

    if model in REASONING_MODELS:
        if model.startswith("qwen/qwen3"):
            # qwen3 family: reasoning_effort="none" TRULY disables thinking
            kwargs["reasoning_effort"] = "none"
        else:
            # gpt-oss family reasons always-on and can't be fully disabled
            kwargs["reasoning_format"] = "hidden"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def chat_json(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> dict:
    raw_response_text = chat(
        prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Clean out any <think> blocks and parse the remaining text as JSON
    clean = _THINK_BLOCK.sub("", raw_response_text).strip()

    if _UNCLOSED_THINK.search(clean):
        raise ValueError(
            "Response contains an unterminated <think> block — the model was "
            "likely truncated mid-reasoning by max_tokens before reaching the "
            "actual answer. Either raise max_tokens substantially for this "
            "model, or switch this call to a non-reasoning model."
        )

    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip().strip("```")

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse response as JSON. Cleaned: {clean} | Raw: {raw_response_text}"
        ) from e