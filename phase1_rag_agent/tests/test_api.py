"""Smoke tests for the FastAPI service (no LLM calls)."""

from fastapi.testclient import TestClient

from src.api import app


def test_health_before_ingest():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["index_loaded"] is False


def test_chat_without_index_returns_409():
    with TestClient(app) as client:
        r = client.post("/chat", json={"question": "hello"})
        assert r.status_code == 409


def test_ingest_missing_file_returns_400():
    with TestClient(app) as client:
        r = client.post("/ingest", json={"pdf_path": "/no/such/file.pdf"})
        assert r.status_code == 400


def test_unknown_session_returns_404():
    with TestClient(app) as client:
        r = client.get("/sessions/does-not-exist")
        assert r.status_code == 404
