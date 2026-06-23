# AI Research Assistant

![CI](https://github.com/NirbhaySharma504/ai-research-assistant/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-orange)

A multi-agent research system that takes a question, plans and runs the web research
on its own, fact-checks what it finds against the sources it pulled, and writes a
fully cited answer. It then grades its own output with RAGAS quality metrics.

## Demo

[![AI Research Assistant demo walkthrough](https://img.youtube.com/vi/QoKrNUio_0M/hqdefault.jpg)](https://youtu.be/QoKrNUio_0M)

[Watch the walkthrough](https://youtu.be/QoKrNUio_0M): the agents report progress one
by one, then you get the final cited answer and the RAGAS self-evaluation scores.

## Why this exists

LLMs sound confident, but they hallucinate, they can't tell you where a claim came
from, and they give you no way to judge whether an answer is actually trustworthy.
Doing the research by hand (searching, opening a dozen tabs, cross-checking facts,
writing a cited summary) is slow and tedious.

This project automates that workflow with a team of specialized agents, and just as
importantly, it measures the quality of its own answers with RAGAS, so reliability is
a number rather than a guess. It is meant to show three things that matter in real AI
engineering: orchestrating multiple agents with explicit control flow and a
verification loop, grounding every answer in retrieved sources with inline citations
(RAG), and evaluating the system properly. That last part includes a fact-checker
ablation and an honest look at where the metrics, and the tooling behind them, fall
short.

It is built on LangGraph with a swappable LLM backend (local Ollama `llama3.1:8b` on
GPU, or hosted Groq `llama-3.3-70b`), a per-session ChromaDB vector store for RAG, a
FastAPI and WebSocket backend that streams live progress, and a Streamlit UI.

## Highlights

- Five specialized agents orchestrated as a stateful LangGraph graph, with a
  research and fact-check loop over prioritized focus areas.
- Retrieval-augmented generation: scraped content is chunked, embedded (MiniLM), and
  stored in a per-session ChromaDB collection. The synthesizer answers only from
  retrieved context and emits inline `[N]` citations.
- Self-evaluation: every answer is scored with RAGAS (faithfulness, answer relevancy,
  context precision), judged by a stronger model (OpenRouter `gpt-4o-mini`) for
  reliable structured output.
- Live progress streamed agent by agent over a WebSocket to the UI.
- Provider-agnostic: switch between local and hosted LLMs with a single env var.
- Persistent history: every run is stored in SQLite and can be replayed.
- Containerized: `docker compose up` brings up the backend and frontend (Ollama stays
  native on the host for GPU access).

## Architecture

```mermaid
graph TD
    Q[User Query] --> P[Planner<br/>decompose into 3-5 focus areas]
    P --> R[Researcher<br/>search, select, scrape, embed, store]
    R --> F[Fact-Checker<br/>extract and verify claims vs sources]
    F -->|more focus areas| R
    F -->|all done| S[Synthesizer<br/>RAG over store, cited answer]
    S --> E[Evaluator<br/>RAGAS scoring]
    E --> A[Cited Answer + Scores]

    R -.-> VS[(ChromaDB<br/>per-session)]
    S -.-> VS
    F -.-> VS
```

| Agent | Role |
|-------|------|
| Planner | Decomposes the query into 3 to 5 prioritized, non-overlapping focus areas (structured JSON). |
| Researcher | For each focus area it generates search queries, runs a Tavily search, has the LLM pick the best URLs, scrapes them asynchronously (trafilatura with a BS4 fallback), then chunks, embeds, and stores them in ChromaDB. |
| Fact-Checker | Extracts verifiable claims from new content and checks each one against the session's stored sources (with a Tavily fallback), then advances the loop. |
| Synthesizer | Runs RAG retrieval over the session store plus the verified claims, and writes a markdown answer with inline `[N]` citations. |
| Evaluator | Scores the answer with RAGAS (faithfulness, answer relevancy, context precision) using a reliable judge LLM (OpenRouter `gpt-4o-mini`, with Groq and Ollama as fallbacks). |

### Tech stack

LangGraph, LangChain 0.3, Ollama and Groq, ChromaDB, sentence-transformers (MiniLM),
Tavily, trafilatura, RAGAS, FastAPI, WebSockets, SQLAlchemy and SQLite, Streamlit,
Docker.

## Evaluation and ablation

Quality isn't asserted here, it is measured. A reproducible harness
(`benchmark/run_eval.py`) runs a fixed question set through the pipeline and averages
the RAGAS scores. It runs two variants to isolate what the fact-checking agent
actually contributes:

- `full`: the complete graph
- `no_factcheck`: the fact-checker swapped for a no-op (the ablation)

The numbers below are generated into [`benchmark/RESULTS.md`](benchmark/RESULTS.md).
Regenerate them with `python -m benchmark.run_eval`.

<!-- RESULTS:START -->
12 questions across 2 variants. The pipeline runs on local `llama3.1:8b`, judged by
OpenRouter `gpt-4o-mini`. The headline figure is the median (which is robust to
outliers); the mean is in parentheses.

| Variant | N | Faithfulness | Answer Relevancy | Context Precision |
|---------|:-:|:-:|:-:|:-:|
| `full` (with fact-checker) | 12 | 0.978 (0.936) | 0.957 (0.774) | 0.974 (0.964) |
| `no_factcheck` (ablation) | 12 | 0.976 (0.908) | 0.834 (0.604) | 0.982 (0.937) |

Controlled ablation: both variants are synthesized and scored over the identical
retrieved corpus (built once per question), so the only thing that changes is the
fact-checker's verified claims (about 13 per answer versus 0). Here the delta is
`full` minus `no_factcheck`:

| Metric | Delta (median) | Delta (mean) |
|--------|:-:|:-:|
| faithfulness | +0.002 | +0.028 |
| answer_relevancy | +0.123 | +0.170 |
| context_precision | -0.008 | +0.027 |

The takeaway: with retrieval held identical, the fact-checking agent gives a small but
consistent improvement. Answer relevancy and faithfulness both go up, and context
precision is unchanged. The verified claims give the synthesizer focused,
cross-checked facts to anchor the answer.

Reading the numbers honestly:

- The relevancy gap looks large by mean (+0.170), but that is inflated by a RAGAS
  artifact: its noncommittal classifier sometimes scores a genuinely relevant answer
  as 0.0 (`no_factcheck` hit 4 of these, `full` hit 2, and all were confirmed on-topic,
  with answers saved in [`results.json`](benchmark/results.json)). Excluding those
  zeros, mean relevancy is 0.929 for `full` versus 0.906 for `no_factcheck`, a +0.023
  effect, real but modest. The robust median difference is +0.123.
- Faithfulness median is 0.978 (6 of 12 queries scored a perfect 1.0), and the
  controlled +0.028 mean is the honest size of the fact-checker's grounding benefit.
- An earlier uncontrolled ablation (two independent pipelines) buried this signal in
  web-retrieval variance. Fixing the experiment design surfaced it.

Full per-query analysis and the saved answers are in
[`benchmark/RESULTS.md`](benchmark/RESULTS.md) and
[`results.json`](benchmark/results.json).
<!-- RESULTS:END -->

A quick glossary: faithfulness means the answer is grounded in the retrieved context,
answer relevancy means the answer addresses the question, and context precision means
the retrieved chunks are on-topic. Context precision is computed over the chunks
actually fed to the synthesizer (after relevance filtering), not whole scraped pages.
See `backend/agents/synthesizer.py`.

```bash
python -m benchmark.run_eval                  # both variants, full question set
python -m benchmark.run_eval --limit 3        # quick smoke run
python -m benchmark.run_eval --report-only    # rebuild the table from cache
```

## Quickstart (local)

Prerequisites: Python 3.11 or 3.12, [Ollama](https://ollama.com), a free
[Tavily API key](https://tavily.com), and a judge LLM for evaluation. For the judge,
an [OpenRouter key](https://openrouter.ai) is recommended (about $0.20 for the full
benchmark); a [Groq key](https://console.groq.com) also works. Without either, RAGAS
falls back to the local model.

```bash
git clone https://github.com/NirbhaySharma504/ai-research-assistant.git
cd ai-research-assistant

python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

cp .env.example .env        # add TAVILY_API_KEY (and OPENROUTER_API_KEY for RAGAS eval)

# Local model (GPU auto-detected on Linux):
ollama pull llama3.1:8b

# Sanity check the full pipeline from the CLI:
python -m scripts.run_research "What are the main causes of climate change?"
```

### Run the app

```bash
# Terminal 1: backend (FastAPI + WebSocket)
uvicorn backend.api.app:app --reload --port 8000

# Terminal 2: frontend (Streamlit)
streamlit run frontend/app.py
```

Open http://localhost:8501. The API docs are at http://localhost:8000/docs.

## Run with Docker

Ollama stays native on the host so it keeps GPU access; the containers reach it via
`host.docker.internal`.

```bash
# On the host:
ollama serve            # if not already running as a service
ollama pull llama3.1:8b

# .env must contain TAVILY_API_KEY (compose reads it):
docker compose up --build
```

Frontend at http://localhost:8501, backend at http://localhost:8000.

## Deployment

The backend needs Ollama and a GPU for local inference, so it doesn't fit free hosting
tiers. The recommended setup is a deployed Streamlit frontend talking to a self-hosted
backend:

1. Run the backend locally (`uvicorn`, or `docker compose up backend`) on your GPU
   machine.
2. Expose it with a tunnel: `cloudflared tunnel --url http://localhost:8000` (or
   `ngrok http 8000`).
3. Deploy `frontend/app.py` to Streamlit Community Cloud and set `BACKEND_URL` (the
   tunnel URL) in the app's Secrets. See
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).

The frontend resolves its backend from `BACKEND_URL` (env var first, then a Streamlit
secret, then localhost), so the same code runs locally and in the cloud without
changes. Because the backend's GPU dependency makes an always-on public demo
impractical on free tiers, the [demo video](https://youtu.be/QoKrNUio_0M) is the
canonical "see it working" artifact, with the local quickstart above as the
interactive path.

## Observability

Every agent step and LLM call can be traced in LangSmith. Set
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` (see
[`.env.example`](.env.example)). LangGraph emits traces automatically, and tracing is
a no-op when unset, so there is no overhead by default. It makes the otherwise opaque
multi-agent loop debuggable: you can see each node's prompt, output, and latency, and
follow the full research, fact-check, synthesize path.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Liveness check. |
| `POST` | `/api/research` | Run a query to completion (blocking). Body: `{query, max_iterations}`. |
| `GET`  | `/api/history` | List past runs. |
| `GET`  | `/api/research/{session_id}` | Fetch a stored run. |
| `WS`   | `/ws/research` | Send `{query, max_iterations}`; receive a `started` event, then `progress` events, then `complete`. |

## Configuration (`.env`)

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `groq` (hosted). This is the pipeline LLM. |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model. |
| `OPENROUTER_API_KEY` | (none) | Preferred RAGAS judge (`gpt-4o-mini`); most reliable. |
| `GROQ_API_KEY` | (none) | Alternative judge, and the hosted LLM-provider swap. |
| `TAVILY_API_KEY` | (none) | Required for web search. |
| `MAX_RESULTS_PER_SEARCH` | `5` | Tavily results per query. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | RAG chunking. |
| `RETRIEVAL_TOP_K` | `8` | Chunks retrieved for the synthesizer. |
| `CONTEXT_MAX_DISTANCE` | `0.70` | Cosine-distance ceiling for a chunk to count as relevant (improves context precision). |

## Project structure

```
backend/
  agents/        planner, researcher, fact_checker, synthesizer, evaluator, utils
  graph/         state, edges, research_graph (assembly + ablation), runner (streaming)
  tools/         search (Tavily), scraper (trafilatura+BS4), vector_store (ChromaDB)
  api/           FastAPI app (REST + WebSocket), schemas
  db/            SQLAlchemy models, crud, database
  config.py      pydantic-settings, and llm.py (provider factory + judge)
  observability.py   opt-in LangSmith tracing
frontend/app.py  Streamlit UI (live progress over WebSocket + history)
benchmark/       run_eval.py (RAGAS harness + ablation), questions, RESULTS.md
scripts/run_research.py   CLI runner
tests/           offline unit tests (pytest)
.github/workflows/ci.yml  CI: install + pytest on every push and pull request
```

## Tests

```bash
pytest tests/ -q     # fast, offline unit tests (JSON parsing, routing, score cleaning)
```

CI runs these on every push and pull request via GitHub Actions.

## Design notes

- Robust JSON from local models: a three-layer parser (strip code fences, then try a
  direct parse, then regex-extract the first JSON block) with a strict-retry nudge
  keeps agent outputs reliable on 8B models.
- Graceful degradation: roughly 20 to 30 percent of URLs fail to scrape (403s,
  timeouts, bot walls), so every scrape path returns an empty string instead of
  raising, and the graph keeps going.
- Non-blocking server: the synchronous graph runs in a worker thread, and progress
  hops back to the event loop via `loop.call_soon_threadsafe` for the WebSocket.
- Per-session isolation: each run gets its own ChromaDB collection, so retrieval never
  crosses sessions.
