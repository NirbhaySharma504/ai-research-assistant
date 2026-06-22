# HANDOFF — AI Research Assistant

This file lets a fresh Claude Code session (on the Ubuntu PC) continue the build.
Read this + `ai_research_assistant_engineering_plan.md` (original design) first.

## Why we moved machines
Started on a Mac with 8 GB RAM (no GPU) — too tight for `llama3.1:8b`. Moving to
Ubuntu with **16 GB RAM + RTX 2060 Super (8 GB VRAM, CUDA)**, where Ollama runs the
8B model on GPU. So the local default model is now `llama3.1:8b` (see `.env.example`).

## Approved build plan
The full plan was written during planning. Key decisions:
- **Swappable LLM provider** via `backend/llm.py` (`LLM_PROVIDER=ollama|groq`). Local =
  Ollama `llama3.1:8b`; Groq = hosted `llama-3.3-70b-versatile`, also used as RAGAS judge.
- **Python 3.11 venv** (NOT 3.14 — torch/chromadb/ragas lack 3.14 wheels).
- **LangChain pinned to the 0.3 line** + **ragas 0.2.x** (langchain 1.x is NOT yet
  compatible with ragas — it breaks on a removed `langchain_community.chat_models.vertexai`
  import). See `requirements.txt` pins — do not loosen them.
- Scraper uses **trafilatura** (primary) + BS4 (fallback); dropped newspaper3k.
- MVP-first: get the LangGraph pipeline running end-to-end via CLI before FastAPI/Streamlit.

## What's DONE (Phases 0-3 code written)
- `requirements.txt`, `.env.example`, `.gitignore`
- `backend/config.py` (pydantic-settings), `backend/llm.py` (provider factory + judge)
- `backend/graph/state.py` — ResearchState (uses `typing_extensions.TypedDict`, required
  by pydantic on Python < 3.12)
- `backend/tools/` — `search.py` (Tavily), `scraper.py` (httpx+trafilatura+BS4),
  `vector_store.py` (ChromaDB per-session, cosine, MiniLM)
- `backend/agents/` — `utils.py` (robust JSON parse + retry), `planner.py`, `researcher.py`,
  `fact_checker.py` (also advances `current_focus_index`), `synthesizer.py`, `evaluator.py`
- `backend/graph/edges.py` (`should_continue`), `backend/graph/research_graph.py`
  (`build_research_graph` + `initial_state`)
- `scripts/run_research.py` — CLI runner (the MVP gate)

### Verified so far
- All deps install on Python 3.11; all imports OK (ragas 0.2.15, EvaluationDataset API).
- Graph compiles. `state.py` fixed to use `typing_extensions.TypedDict`.

### NOT yet verified (do this first on the PC)
- **End-to-end CLI run** — needs a `TAVILY_API_KEY` in `.env` (free tier at tavily.com).
  This is the Phase 3 gate. Run:
  `python -m scripts.run_research "What are the main causes of climate change?"`
- Planner JSON reliability with `llama3.1:8b` (was about to test on Mac with 3.2:3b).
- RAGAS scores non-NaN (set `GROQ_API_KEY` for the judge, or it falls back to Ollama).

## Remaining phases (not started)
- Phase 4: confirm `evaluator.py` produces real scores.
- Phase 5: FastAPI routes + WebSocket (`backend/api/`), SQLite/SQLAlchemy (`backend/db/`).
  WebSocket should emit progress from inside nodes via an `asyncio.Queue`; run the graph
  off the event loop so it doesn't block FastAPI.
- Phase 6: Streamlit `frontend/app.py`.
- Phase 7: README (arch diagram, demo GIF, RAGAS table), Docker (backend+frontend only,
  Ollama stays native on host).

## Setup on the Ubuntu PC
```bash
git clone <your-repo-url> && cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate   # use 3.11 or 3.12, NOT 3.14
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env        # then add TAVILY_API_KEY (and optionally GROQ_API_KEY)
# Ollama (Linux installs with GPU/CUDA support automatically):
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
# sanity:
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # expect True
nvidia-smi   # confirm the GPU is visible
# MVP gate:
python -m scripts.run_research "What are the main causes of climate change?"
```

## Gotchas learned
- `ollama serve` must be running before `ollama pull`/calls (Linux service usually
  auto-starts; on Mac it didn't and a pull silently no-op'd).
- Pydantic on Python 3.11 requires `typing_extensions.TypedDict`, not `typing.TypedDict`.
- Keep langchain on 0.3.x; ragas 0.2.x is incompatible with langchain 1.x.
