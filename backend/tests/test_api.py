import os

os.environ.setdefault("PELES_MODEL_BACKEND", "heuristic")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["active_backend"] == "heuristic"


def test_evaluate_returns_a_named_winner():
    resp = client.post(
        "/api/evaluate",
        json={
            "model_a_name": "GPT-4o",
            "model_b_name": "Claude 3.5 Sonnet",
            "prompt": "What is the capital of France?",
            "response_a": "The capital of France is Paris, a city on the Seine known for the Eiffel Tower.",
            "response_b": "idk",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["winner_name"] in {"GPT-4o", "Claude 3.5 Sonnet", "Tie"}
    assert abs(body["prob_model_a"] + body["prob_model_b"] + body["prob_tie"] - 1.0) < 1e-3
    assert body["backend"] == "heuristic"


def test_evaluate_rejects_empty_response():
    resp = client.post(
        "/api/evaluate",
        json={
            "model_a_name": "A",
            "model_b_name": "B",
            "response_a": "",
            "response_b": "fine",
        },
    )
    assert resp.status_code == 422
