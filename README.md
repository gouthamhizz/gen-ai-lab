# GenAI Platform — RAG + Agent System (12-Step Capstone)

A cumulative, from-scratch build of a production-shaped Retrieval-Augmented Generation and Agent
platform, ending in a concurrent FastAPI service with SSE token streaming. Each step's code lives in
`src/genai_platform/` and is exercised standalone by the matching script in `steps/`; the final
artifact is the whole service, deployed with Docker Compose and monitored with Prometheus/Grafana +
Langfuse.

## Stack at a glance

| Concern            | Choice                                                            |
|---------------------|--------------------------------------------------------------------|
| Local LLM           | Llama-3 8B via Ollama                                              |
| Cloud LLM / fallback| Google Gemini (`google-genai` SDK) — also reachable via its OpenAI-compatible endpoint |
| Embeddings          | `sentence-transformers/all-MiniLM-L6-v2` (local) or Gemini embeddings |
| Vector DB           | Qdrant                                                             |
| Keyword search      | BM25 (`rank_bm25`), fused with dense search via Reciprocal Rank Fusion |
| Conversation memory | Redis, with LLM-compressed rolling summaries                      |
| Agent framework     | Hand-written ReAct loop; CrewAI for the multi-agent report         |
| Evaluation          | RAGAS (Context Precision/Recall, Faithfulness, Answer Relevancy)   |
| Tracing             | Langfuse (self-hosted)                                             |
| Metrics             | Prometheus + Grafana                                                |
| API                 | FastAPI + Uvicorn + Server-Sent Events                             |

## Repository layout

```
src/genai_platform/
  llm/            Step 1  — provider gateway (Ollama/Gemini/OpenAI-compat) + LangChain adapter
  embeddings/     Step 2  — hand-written + vectorised cosine/Euclidean maths, embedding backends
  vectorstore/    Step 3  — Qdrant collection schema, HNSW config, search
  ingestion/      Step 4  — PDF/HTML loaders, recursive + semantic chunking, ingest pipeline
  rag/            Step 5-6— prompts, dense/hybrid retrievers, LCEL RAG chain
  memory/         Step 7  — Redis conversation buffer with summary compression
  agents/         Step 8-9— ReAct agent + tools, CrewAI Researcher/Writer/Reviewer crew
  evaluation/     Step 10 — RAGAS harness
  observability/  Step 11 — Prometheus metrics, Langfuse tracing
  api/            Step 12 — FastAPI routes, SSE, auth, dependency wiring
steps/            One runnable demo script per step (01-12)
tests/            pytest unit tests (34 tests, no live LLM/network needed)
loadtest/         Locust script demonstrating concurrency
monitoring/       Prometheus + Grafana provisioning
docs/             Per-step Word walkthrough documents
```

## Prerequisites (Windows + WSL2, Python 3.12)

1. **WSL2** with a Linux distro, updated (`wsl --update` from PowerShell, `sudo apt update && sudo apt upgrade` inside).
2. **Docker Desktop**, with *Settings → Resources → WSL Integration* enabled for your distro.
3. **Ollama** already installed with Llama-3 8B pulled. Confirm the exact tag:
   ```bash
   ollama list
   ```
   Copy whatever tag you see (e.g. `llama3:8b` or `llama3.1:8b`) into `.env` as `OLLAMA_MODEL`.
4. A **Gemini API key** (used for the fallback LLM path, Gemini embeddings if selected, and RAGAS's
   judge model when you don't want to hammer your local model). Get one from Google AI Studio.

## Quick start

```bash
git clone <this repo> && cd genai-platform      # or unzip the deliverable
cp .env.example .env                             # then edit: OLLAMA_MODEL, GEMINI_API_KEY
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# unit tests (no Docker/network needed — hash embeddings + in-memory Qdrant + fakeredis)
export PYTHONPATH=src
pytest -q

# full stack
docker compose --profile full up -d --build
docker compose exec ollama ollama pull llama3.1:8b   # only if Ollama isn't already on your host

curl http://localhost:8000/health
```

### Ingest your own documents

Drop PDFs/HTML into `data/raw/`, then:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"path": "data/raw", "strategy": "recursive", "recreate_collection": true}'
```

### Chat (SSE token streaming)

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"message": "What is the refund policy?", "session_id": "demo-1"}'
```

Events, in order: `sources` (numbered passages with citations) → repeated `token` → `citations`
(the grounding guard's verdict) → `metrics` → `done`.

### Agent (SSE, with reasoning trace)

```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"message": "What is 12 squared, plus the number of moons of Jupiter?", "session_id": "demo-1"}'
```

Events: `thought` / `action` / `observation` (repeated per ReAct step) → `token` (final answer) → `final`.

### Multi-agent report (CrewAI, not streamed — can take minutes)

```bash
curl -X POST http://localhost:8000/crew/report \
  -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"topic": "Company refund and warranty policy"}'
```

### Evaluate the pipeline (RAGAS)

```bash
curl -X POST http://localhost:8000/eval/ragas \
  -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" -d '{"k": 5}'
```

### Load test (concurrency demonstration)

```bash
pip install locust
locust -f loadtest/locustfile.py --host http://localhost:8000
# open http://localhost:8089, set users e.g. 50, spawn rate 5
```

Watch it live: Prometheus at `:9090`, Grafana at `:3001` (admin/admin), Langfuse at `:3000`.

## Running one step at a time

```bash
cd steps
export PYTHONPATH=../src
python step01_llm_gateway.py     # local Llama-3 + retry/backoff demo
python step02_embedding_math.py  # hand-written vs NumPy cosine/Euclidean
python step03_qdrant_setup.py    # collection + HNSW + timed search
python step04_ingest.py          # PDF/HTML -> chunks.jsonl
python step05_basic_rag.py       # cited, dense-only RAG
python step06_hybrid_rag.py      # multi-query + BM25/dense hybrid
python step07_memory_chat.py     # Redis summary-compressed memory
python step08_react_agent.py     # ReAct agent with 3 tools
python step09_crew_report.py     # Researcher/Writer/Reviewer report
python step10_ragas_eval.py      # quantitative RAG evaluation
python step11_tracing_demo.py    # Langfuse trace + Prometheus telemetry
python step12_start_api.py       # prints how to launch the final service
```

## Known compatibility fix (worth knowing about)

Every current RAGAS release unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI`, which modern `langchain-community` no
longer ships (Vertex AI moved to its own package). Since this project never uses Vertex AI,
`evaluation/ragas_eval.py` registers a harmless stub module before importing `ragas`, so `pip
install -r requirements.txt` + `pytest`/`step10` just work without pinning fragile old versions.

## Cloud deployment (after local is working)

The API is a single stateless container reading everything from environment variables, so moving it
is mostly "point `QDRANT_URL` / `REDIS_URL` / `OLLAMA_HOST` at managed or self-hosted instances of
each": e.g. a small VM or a managed Postgres-backed Qdrant Cloud cluster + a managed Redis + Ollama on
a GPU VM (or drop Ollama and set `LLM_PROVIDER=gemini`), then `docker build` and run the same image
behind a load balancer with 2+ replicas. Kubernetes manifests are a natural next step (a Deployment
for `api` with an HPA on CPU/RPS, a StatefulSet or managed service for Qdrant/Redis) — intentionally
left out of this deliverable per the step-by-step plan; ask and we can add them next.
