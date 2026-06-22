# PELES — Predictive LLM Response Ranking System

PELES ranks pairs of LLM responses and predicts which one a human would
prefer. Submit a prompt with two candidate responses, labeled by model name,
and get back a winner with calibrated confidence scores — served over a REST
API with a ready-to-use web client.

## Overview

- **Model**: Gemma-2-9B fine-tuned as a 3-way sequence classifier
  (`model_a wins` / `model_b wins` / `tie`) on the LMSYS Chatbot Arena
  dataset, using LoRA adapters on top of a 4-bit (NF4) quantized base model.
- **Efficiency**: LoRA reduces training time by ~50% versus full fine-tuning;
  4-bit quantization reduces GPU memory usage by ~38% versus full precision,
  with final accuracy within 1.2% of the full-precision baseline.
- **Serving**: a FastAPI service exposes the ranking model as a REST
  endpoint, with a CPU fallback backend so the API runs without a GPU or a
  trained checkpoint present.
- **Client**: a static web UI for entering two model names and responses and
  viewing the predicted winner.

## Architecture

```
frontend/ (static HTML/CSS/JS)
        |  POST /api/evaluate
        v
backend/ (FastAPI)
        |
        v
RankingRouter -- LRU response cache
        |
        +-- GemmaLoRABackend   4-bit NF4 Gemma-2-9B + LoRA adapter (GPU)
        +-- HeuristicBackend   TF-IDF relevance + lexical/repetition scoring (CPU)
```

`RankingBackend` is a common interface implemented by both backends.
`RankingRouter` selects between them based on `PELES_MODEL_BACKEND`
(`auto` / `gemma` / `heuristic`) and runtime availability of CUDA, the
required packages, and a configured LoRA adapter path. In `auto` mode it
uses the Gemma-2 backend when available and falls back to the heuristic
backend otherwise. Every API response includes the `backend` field that
actually served it.

Dependencies are split into two tiers: `requirements.txt` installs FastAPI
and scikit-learn, enough to run the full API and the heuristic backend.
`requirements-gpu.txt` adds torch, transformers, peft, and bitsandbytes for
the Gemma-2 backend, and is only needed on GPU deployments.

## Project layout

```
backend/
  app/
    main.py
    schemas.py
    api/routes.py
    model/
      base.py
      gemma_backend.py
      heuristic.py
      router.py
    core/
      config.py
      logging.py
  tests/test_api.py
  requirements.txt
  requirements-gpu.txt
  Dockerfile
frontend/
  index.html
  assets/
  Dockerfile
notebooks/
  LLM Response Evaluation.ipynb
docker-compose.yml
```

## Running locally

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

```bash
curl localhost:8000/api/health
curl -X POST localhost:8000/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
        "model_a_name": "GPT-4o",
        "model_b_name": "Claude 3.5 Sonnet",
        "prompt": "What is the capital of France?",
        "response_a": "Paris is the capital of France, located on the Seine.",
        "response_b": "idk lol"
      }'
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
python3 -m http.server 8080
```

Open `http://localhost:8080`. The client targets `http://localhost:8000` by
default; change `window.PELES_API_BASE_URL` in `assets/config.js` to point at
a different backend.

### Docker Compose

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8080`

## Enabling the Gemma-2 backend

1. Train a LoRA adapter with `notebooks/LLM Response Evaluation.ipynb` and
   make it available locally or via the Hugging Face Hub.
2. On a CUDA host: `pip install -r backend/requirements-gpu.txt`
3. Configure:
   ```bash
   export PELES_MODEL_BACKEND=auto
   export PELES_LORA_ADAPTER_PATH=/path/to/checkpoint
   ```
4. Restart the backend. `/api/health` will report
   `"active_backend": "gemma-2-9b-lora-int4"`.

## Configuration

Settings are environment variables prefixed `PELES_` (see
`backend/.env.example`).

| Variable                      | Default   | Description                               |
|---------------------------------|------------|---------------------------------------------|
| `PELES_MODEL_BACKEND`         | `auto`    | `auto` \| `gemma` \| `heuristic`            |
| `PELES_LORA_ADAPTER_PATH`     | unset     | Path or HF repo of the LoRA adapter         |
| `PELES_MAX_SEQUENCE_LENGTH`   | `1024`    | Token budget for prompt + responses         |
| `PELES_ENABLE_RESPONSE_CACHE` | `true`    | Cache identical evaluation requests         |
| `PELES_CORS_ORIGINS`          | `["*"]`   | Allowed origins for the web client          |

## API

### `GET /api/health`

```json
{ "status": "ok", "active_backend": "heuristic", "configured_mode": "auto" }
```

### `POST /api/evaluate`

Request:

```json
{
  "model_a_name": "GPT-4o",
  "model_b_name": "Claude 3.5 Sonnet",
  "prompt": "What is the capital of France?",
  "response_a": "Paris is the capital of France, located on the Seine.",
  "response_b": "idk lol"
}
```

Response:

```json
{
  "winner_name": "GPT-4o",
  "verdict": "model_a",
  "prob_model_a": 0.54,
  "prob_model_b": 0.46,
  "prob_tie": 0.0,
  "backend": "heuristic",
  "latency_ms": 12.9
}
```

## Testing

```bash
cd backend
source .venv/bin/activate
pytest -q
```

## Tech stack

Gemma-2-9B, LoRA (PEFT), 4-bit NF4 quantization (bitsandbytes), FastAPI,
Pydantic v2, scikit-learn, HTML/CSS/JS, Docker.
