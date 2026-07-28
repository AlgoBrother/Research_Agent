import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from data_pipeline.router.router import QueryRouter

load_dotenv()

_client: OpenAI | None = None  # Lazy initialization

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

def get_router(client: OpenAI | None = None) -> QueryRouter:
    """Instantiates and returns QueryRouter using the shared client."""
    if client is None:
        client = get_client()
    return QueryRouter(client=client, model="llama-3.1-8b-instant")

def chat(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    max_tokens: int = 1024
) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,   
        max_tokens=max_tokens,       
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
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
    clean = raw_response_text.strip()
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