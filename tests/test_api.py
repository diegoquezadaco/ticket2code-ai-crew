"""Tests for HTTP endpoints, streamed chat responses, and project downloads."""

import io
import json
import zipfile

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


def test_project_download_contains_every_session_file():
    session_id = "project-download-test"
    main.store.get_or_create(session_id).files = {
        "src/main.py": "print('hello')\n",
        "README.md": "# Project\n",
    }

    download = TestClient(main.app).get(
        f"/api/sessions/{session_id}/download"
    )
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert set(archive.namelist()) == {"src/main.py", "README.md"}
        assert archive.read("src/main.py").decode() == "print('hello')\n"


def test_invalid_session_id_is_rejected():
    r = TestClient(main.app).post("/api/chat", json={"session_id": "short", "message": "hi"})
    assert r.status_code == 422


def test_unknown_trace_is_404():
    assert TestClient(main.app).get("/api/sessions/nope-nope-nope/trace").status_code == 404


def test_project_download_requires_generated_files():
    main.store.get_or_create("empty-project-test").files.clear()
    response = TestClient(main.app).get("/api/sessions/empty-project-test/download")
    assert response.status_code == 404
