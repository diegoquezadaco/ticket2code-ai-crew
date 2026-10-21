import json

from fastapi.testclient import TestClient

from app import main

from .fakes import APPROVE, GOOD_CODE, PLAN, FakeLLM, route


def test_health():
    assert TestClient(main.app).get("/health").json()["ok"] is True


def test_chat_streams_logs_then_final_and_exposes_trace(monkeypatch):
    fake = FakeLLM({"router": [route()], "planner": [PLAN], "coder": [GOOD_CODE], "reviewer": [APPROVE]})
    monkeypatch.setattr(main, "llm", fake)
    client = TestClient(main.app)
    body = {"session_id": "session-1234", "message": "add coupon validation"}
    with client.stream("POST", "/api/chat", json=body) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    types = [e["type"] for e in events]
    assert types[-1] == "final" and types.count("log") >= 4
    assert {e["agent"] for e in events if e["type"] == "log"} >= {"router", "planner", "coder", "reviewer"}
    trace = client.get("/api/sessions/session-1234/trace").json()
    assert trace["runs"][0]["request"] == "add coupon validation"


def test_invalid_session_id_is_rejected():
    r = TestClient(main.app).post("/api/chat", json={"session_id": "short", "message": "hi"})
    assert r.status_code == 422


def test_unknown_trace_is_404():
    assert TestClient(main.app).get("/api/sessions/nope-nope-nope/trace").status_code == 404
