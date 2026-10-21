"""FastAPI app: chat endpoint (streamed), trace endpoint, health check and the static front end."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .llm import LLMClient
from .pipeline import run_turn
from .session import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC = Path(__file__).resolve().parent.parent / "static"
settings = get_settings()
llm = LLMClient(settings)
store = SessionStore(settings.max_sessions)
app = FastAPI(title="Ticket to Code: multi-agent pipeline")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"ok": True, "llm_configured": llm.configured, "max_retries": settings.max_retries}


@app.post("/api/chat")
def chat(body: ChatIn):
    """Streams newline-delimited JSON events: log* then exactly one of final | error."""
    session = store.get_or_create(body.session_id)

    def stream():
        if not session.lock.acquire(blocking=False):
            yield json.dumps({"type": "error", "stage": "server",
                              "message": "This conversation is still processing the previous message."}) + "\n"
            return
        try:
            for event in run_turn(session, body.message.strip(), llm, settings):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            session.lock.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sessions/{session_id}/trace")
def trace(session_id: str):
    """Full agent trace of every turn in a conversation (also shown live in the UI)."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    return {"session_id": session_id, "runs": session.runs}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

