"""
api.py — HTTP entry point for ArXAgent, used by the Next.js UI.

Run locally:
    export ARXAGENT_BACKEND_SECRET=$(openssl rand -hex 32)   # or put in .env
    uvicorn api:app --reload --port 8000

Then in arxagent-ui/.env.local:
    ARXAGENT_BACKEND_URL=http://localhost:8000
    ARXAGENT_BACKEND_SECRET=<same value as above>
"""

import asyncio
import json
import os
import queue
import threading

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main_chat import ResearchAgent
from rate_limiter import limiter

app = FastAPI()

BACKEND_SECRET = os.environ.get("ARXAGENT_BACKEND_SECRET")

# One ResearchAgent per session, kept in memory. ResearchAgent carries
# conversation history, session memory, and _last_papers per user, so it
# has to be a stateful instance per session — not a single shared one.
# Single-process only (matches a single Render/Fly free-tier instance).
_agents: dict[str, ResearchAgent] = {}
_agents_lock = threading.Lock()


def get_agent(session_id: str) -> ResearchAgent:
    with _agents_lock:
        agent = _agents.get(session_id)
        if agent is None:
            agent = ResearchAgent(top_k=6)
            _agents[session_id] = agent
        return agent


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


def check_auth(authorization: str | None):
    if not BACKEND_SECRET:
        raise HTTPException(500, "ARXAGENT_BACKEND_SECRET not set on server")
    if authorization != f"Bearer {BACKEND_SECRET}":
        raise HTTPException(401, "Unauthorized")


def serialize_paper(p) -> dict:
    published = getattr(p, "published", None)
    return {
        "title": getattr(p, "title", None),
        "pdf_url": getattr(p, "pdf_url", None),
        "authors": getattr(p, "authors", []),
        "published": published.isoformat() if published else None,
        "summary": getattr(p, "summary", None),
        "arxiv_id": getattr(p, "arxiv_id", None),
    }


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/query")
async def query(req: QueryRequest, request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)

    # Fall back to client IP if the frontend hasn't assigned a session id yet.
    session_id = req.session_id or (request.client.host if request.client else "anonymous")

    allowed, retry_after = limiter.check(session_id)
    if not allowed:
        raise HTTPException(
            429,
            detail=f"Rate limit exceeded. Try again in {retry_after:.0f}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    agent = get_agent(session_id)
    q: "queue.Queue" = queue.Queue()
    DONE = object()

    def on_step(msg: str):
        q.put(("step", msg))

    def run_agent():
        try:
            result = agent.ask(req.query, on_step=on_step)
            q.put(("answer", result))
        except Exception as e:  # surface the error to the client instead of hanging
            q.put(("error", str(e)))
        finally:
            q.put((DONE, None))

    threading.Thread(target=run_agent, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            kind, payload = await loop.run_in_executor(None, q.get)
            if kind is DONE:
                break
            if kind == "step":
                yield sse({"type": "step", "message": payload})
            elif kind == "answer":
                yield sse({
                    "type": "answer",
                    "answer": payload["answer"],
                    "papers": [serialize_paper(p) for p in payload["papers"]],
                })
            elif kind == "error":
                yield sse({"type": "error", "message": payload})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}