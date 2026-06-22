# AI Research Assistant — Full Engineering Plan

---

## 1. WHAT WE ARE BUILDING

A multi-agent AI system that takes a research query, autonomously breaks it into
focus areas, searches the web, scrapes content, fact-checks key claims, and
synthesizes a comprehensive cited answer. Every component is measurable (RAGAS),
deployable (Docker), and explainable in an interview.

CV headline:
"Built a multi-agent AI research assistant using LangGraph with 4 specialized agents,
RAG pipeline with ChromaDB, RAGAS evaluation, deployed via FastAPI + Streamlit"

---

## 2. FUNCTIONAL REQUIREMENTS

FR-01  User submits a research query via Streamlit UI
FR-02  System generates 3-5 prioritized research focus areas (Planner agent)
FR-03  For each focus area, system searches the web and retrieves relevant content (Researcher agent)
FR-04  Retrieved content is chunked, embedded, and stored in ChromaDB
FR-05  Key claims in retrieved content are cross-verified against multiple sources (Fact-checker agent)
FR-06  All verified content is synthesized into a cited markdown answer (Synthesizer agent)
FR-07  Research progress is streamed in real-time via WebSocket to the frontend
FR-08  Final output includes answer + numbered citation markers + source URLs
FR-09  System evaluates output quality using RAGAS after synthesis
FR-10  User can ask follow-up questions about research findings (using session RAG context)
FR-11  Research sessions are persisted in SQLite for retrieval

---

## 3. NON-FUNCTIONAL REQUIREMENTS

NFR-01  Entire system runs locally — no paid APIs required (Ollama for LLM)
NFR-02  Research session completes within 3-5 minutes for a typical query
NFR-03  Each LLM call has a 60-second timeout with graceful error handling
NFR-04  System handles concurrent research sessions (FastAPI async)
NFR-05  ChromaDB uses cosine similarity for semantic retrieval
NFR-06  RAGAS faithfulness score target: >0.70 (baseline, iterate to improve)
NFR-07  All API endpoints auto-documented via FastAPI Swagger
NFR-08  Docker Compose brings up all services with one command
NFR-09  No hardcoded secrets — all config via .env file

---

## 4. TECHNOLOGY STACK — EVERY DECISION JUSTIFIED

### LLM: Ollama + Llama 3.1 8B

  Why Ollama?
    Free, local, no API key needed, no risk of key leaking in public GitHub repo.
    Shows understanding of local LLM deployment — interviewers ask about this.

  Why Llama 3.1 8B specifically?
    128K context window (critical — research sessions accumulate lots of text).
    Strong instruction following. Runs on 8GB RAM (student laptop friendly).
    Smaller models (3B, 7B) struggle with structured JSON output required by agents.

  Why not OpenAI / Gemini?
    OpenAI costs money. Free Gemini tier has rate limits that break long research sessions.
    For a portfolio project, local = reproducible by any recruiter who runs your code.

### Agent Framework: LangGraph

  Why not plain LangChain chains?
    LangChain LCEL is linear (A→B→C). Research requires cycles: after fact-checking,
    if more focus areas remain, you loop back to the Researcher. Linear chains cannot do this.

  Why not CrewAI or AutoGen?
    Less control over state transitions. Harder to add custom logic.
    LangGraph is what Google and Anthropic engineers actually use for production agents.
    Interviewers from these companies recognize LangGraph immediately.

  Why LangGraph wins?
    Explicit StateGraph with typed state (TypedDict).
    Conditional edges for routing decisions.
    Built-in support for cycles (research loop).
    State is fully inspectable at every step — great for debugging.

### Vector Database: ChromaDB

  Why not Pinecone?
    Free tier is very limited. Requires cloud account. Data leaves your machine.

  Why not Weaviate or Qdrant?
    More complex setup. Overkill for a student project.

  Why ChromaDB?
    Free, local, Python-native. PersistentClient stores to disk between sessions.
    Works natively with sentence-transformers embeddings. Zero cloud dependency.

### Embedding Model: all-MiniLM-L6-v2

  Why?
    Free. Runs on CPU. 384-dimensional embeddings. Fast (100ms per batch).
    Available via sentence-transformers (pip install).
    Excellent semantic search quality for retrieval tasks. No API key.

### Web Search: Tavily API

  Why not DuckDuckGo?
    DuckDuckGo's unofficial Python library is rate-limited aggressively
    and returns raw web results, not cleaned content.

  Why Tavily?
    Built specifically for AI agents. Returns pre-cleaned content (not raw HTML).
    Structured JSON response. Free tier: 1000 searches/month — enough for development.
    Has include_raw_content=True flag which saves you a scraping step on many URLs.

  Fallback: if Tavily quota runs out, swap to duckduckgo-search library.

### Web Scraper: httpx + BeautifulSoup4

  Why httpx over requests?
    httpx is async — essential for FastAPI which runs in an async context.
    Requests is synchronous and will block the FastAPI event loop.

  Why BS4?
    Simple, well-documented, sufficient for extracting clean text from any webpage.
    Added: newspaper3k for news article URLs (handles pagination, bylines, etc.)

### Backend: FastAPI

  Why not Flask or Django?
    Flask is synchronous by default. Django is too heavy.
    FastAPI is async-native, has WebSocket support built in,
    auto-generates Swagger docs, and is what modern Python backend teams use.

### Frontend: Streamlit

  Why not React?
    React frontend would take 3-4 weeks to build properly.
    Streamlit gives a working, demonstrable UI in 2-3 days.
    For an internship project, a live demo matters more than frontend framework choice.
    You can honestly say "I used Streamlit to focus engineering effort on the AI pipeline"
    — that is a valid and mature engineering decision.

### Evaluation: RAGAS

  Why?
    Industry-standard framework for evaluating RAG systems.
    Measures exactly what matters: are answers grounded in sources? are they relevant?
    Gives you numeric scores to put on your CV (e.g., "achieved 0.81 faithfulness").
    Recruiters and interviewers recognize RAGAS immediately in 2025-2026.

### Session Storage: SQLite

  Why not PostgreSQL?
    Overkill for a single-user student project.
    SQLite requires zero setup, stores as a single file, perfect for local development.
    Swap to PostgreSQL in 1 line if scaling is needed.

---

## 5. LANGGRAPH STATE DESIGN (THE BACKBONE)

This is the most important design decision in the entire project.
Every agent reads from and writes to this shared state.

```python
# backend/graph/state.py

from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator

class FocusArea(TypedDict):
    title: str           # "Current research landscape in X"
    description: str     # What to look for (2-3 sentences)
    priority: int        # 1-5, agent tackles highest priority first
    status: str          # "pending" | "in_progress" | "completed"

class RetrievedContent(TypedDict):
    url: str
    title: str
    content: str         # cleaned text, capped at 8000 chars
    focus_area: str      # which focus area this was retrieved for
    timestamp: str
    relevance_score: float

class VerifiedClaim(TypedDict):
    claim: str
    is_verified: bool
    confidence: float           # 0.0 to 1.0
    supporting_sources: List[str]
    contradicting_sources: List[str]

class ResearchState(TypedDict):
    # --- INPUT ---
    query: str
    session_id: str

    # --- PLANNER OUTPUT ---
    focus_areas: List[FocusArea]
    current_focus_index: int    # tracks which focus area we are on

    # --- RESEARCHER OUTPUT ---
    # Annotated with operator.add means each agent appends to this list
    # rather than overwriting it. Critical for accumulation across iterations.
    retrieved_content: Annotated[List[RetrievedContent], operator.add]
    search_queries_used: List[str]   # deduplicate queries across iterations

    # --- FACT-CHECKER OUTPUT ---
    verified_claims: List[VerifiedClaim]

    # --- SYNTHESIZER OUTPUT ---
    final_answer: str            # markdown formatted
    citations: List[dict]        # [{number, url, title, quote}]

    # --- EVALUATION ---
    ragas_scores: Optional[dict] # {"faithfulness": 0.81, "answer_relevance": 0.79, ...}

    # --- CONTROL ---
    messages: Annotated[List[BaseMessage], operator.add]
    iteration_count: int
    max_iterations: int          # default 3, set by user
    errors: List[str]            # non-fatal errors get logged here
    status: str                  # "planning"|"researching"|"fact_checking"|"synthesizing"|"evaluating"|"done"|"error"
```

Why operator.add on retrieved_content and messages?
  LangGraph uses this annotation to merge state updates from parallel/cyclic nodes.
  Without it, each agent call would OVERWRITE the list instead of appending.
  This is the most common LangGraph mistake beginners make.

---

## 6. LANGGRAPH GRAPH DESIGN (THE FLOW)

```
START
  |
  v
[planner_node]
  |
  v
[researcher_node] <─────────────┐
  |                              │
  v                              │ (if more focus areas remain)
[fact_checker_node]              │
  |                              │
  v                              │
[should_continue?] ─────────────┘
  |
  | (all focus areas done)
  v
[synthesizer_node]
  |
  v
[evaluator_node]
  |
  v
END
```

Conditional edge function (should_continue):

```python
# backend/graph/edges.py

def should_continue(state: ResearchState) -> str:
    next_index = state["current_focus_index"] + 1
    all_done = next_index >= len(state["focus_areas"])
    over_limit = state["iteration_count"] >= state["max_iterations"]
    has_error = state["status"] == "error"

    if has_error or all_done or over_limit:
        return "synthesize"
    else:
        return "continue_research"
```

Graph assembly:

```python
# backend/graph/research_graph.py

from langgraph.graph import StateGraph, END
from .state import ResearchState
from ..agents.planner import planner_node
from ..agents.researcher import researcher_node
from ..agents.fact_checker import fact_checker_node
from ..agents.synthesizer import synthesizer_node
from ..agents.evaluator import evaluator_node
from .edges import should_continue

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "fact_checker")
    graph.add_conditional_edges(
        "fact_checker",
        should_continue,
        {
            "continue_research": "researcher",
            "synthesize": "synthesizer"
        }
    )
    graph.add_edge("synthesizer", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()
```

---

## 7. EACH AGENT IN FULL DETAIL

---

### AGENT 1: PLANNER

File: backend/agents/planner.py

Input from state:  query
Output to state:   focus_areas (list), status = "researching"

Responsibility:
  Analyze the query and generate 3-5 specific, non-overlapping research
  focus areas. Each focus area targets a different dimension of the query.

System prompt (use this exactly, iterate on it):
```
You are a research planning expert. Given a user research query, generate exactly
3 to 5 research focus areas that together would comprehensively answer the query.

Rules:
- Each focus area must be specific and non-overlapping with others
- Cover different dimensions: background/history, current state, technical details,
  applications, criticism/limitations (pick the most relevant)
- Each description must state EXACTLY what information to look for
- Priority 5 = directly answers the core question, Priority 1 = background context

Return ONLY valid JSON with NO additional text:
{
  "focus_areas": [
    {
      "title": "short descriptive title (5-8 words)",
      "description": "what to search for and why (2-3 sentences)",
      "priority": 5
    }
  ]
}
```

Implementation steps:
  1. Create ChatOllama instance (model="llama3.1:8b", temperature=0.3)
     Lower temperature = more consistent JSON output, less hallucination
  2. Pass system prompt + query to LLM
  3. Parse JSON response with try/except
  4. Validate with Pydantic model (raises clear errors if LLM output is malformed)
  5. Sort focus_areas by priority descending
  6. Return updated state

Common failure mode: LLM wraps JSON in markdown code blocks (```json ... ```)
Fix: strip ```json and ``` before parsing

---

### AGENT 2: RESEARCHER

File: backend/agents/researcher.py

Input from state:  focus_areas[current_focus_index], search_queries_used
Output to state:   retrieved_content (appended), search_queries_used (updated),
                   current_focus_index (incremented), iteration_count (incremented)

Responsibility:
  For the current focus area, generate targeted search queries, retrieve relevant
  web content, scrape it, chunk it, embed it, and store it in ChromaDB.

Step-by-step flow inside researcher_node:

  Step 1 — Generate search queries
    Prompt: "Given this research focus area: [title + description], generate
    2-3 specific Google search queries that would find the most relevant information.
    Return as JSON array of strings."
    Deduplicate against state["search_queries_used"] to avoid repeating searches.

  Step 2 — Execute searches
    For each query: call search_tool(query, max_results=5)
    Collect all results: [{url, title, content, score}]

  Step 3 — Select best URLs
    Prompt: "Given these search results and the focus area description, select the
    2-3 most relevant URLs. Return as JSON array of URLs."
    This uses the LLM to rank relevance rather than just taking top results.

  Step 4 — Scrape selected URLs
    For each URL: call scrape_tool(url)
    Skip URLs that return empty content or error.

  Step 5 — Chunk, embed, store
    For each scraped page: call chunk_and_store(content, metadata, session_id)
    metadata includes: url, title, focus_area, timestamp

  Step 6 — Update state
    Append new RetrievedContent objects to retrieved_content
    Add used queries to search_queries_used
    Increment iteration_count
    Update focus_areas[current_focus_index].status = "completed"

---

### AGENT 3: FACT-CHECKER

File: backend/agents/fact_checker.py

Input from state:  retrieved_content (most recent batch), query
Output to state:   verified_claims (list)

Responsibility:
  Extract key factual claims from retrieved content and verify each
  against multiple sources. Flag low-confidence claims.

Step-by-step flow:

  Step 1 — Extract claims
    Take last N retrieved_content items (N = content from current focus area)
    Prompt: "Extract 5-8 specific, verifiable factual claims from this content.
    Focus on statistics, dates, names, and causal relationships.
    Return as JSON array of strings."

  Step 2 — Verify each claim
    For each claim:
      a. Check if claim appears in 2+ retrieved sources (using ChromaDB similarity search)
         query = claim text, retrieve top 5 chunks, check how many unique sources
      b. If found in 2+ sources: is_verified=True, confidence = 0.8-1.0
      c. If found in 1 source: is_verified=True, confidence = 0.5-0.7
      d. If found in 0 sources from ChromaDB: do 1 verification search via Tavily
         If Tavily confirms: is_verified=True, confidence = 0.6
         If not found: is_verified=False, confidence = 0.2

  Step 3 — Return verified_claims
    Only pass claims with confidence >= 0.5 to Synthesizer
    Log unverified claims to state["errors"] (non-fatal)

Note: Fact-checker is the simplest agent. Keep it fast.
Don't over-engineer the verification. Two source matches is sufficient for a student project.

---

### AGENT 4: SYNTHESIZER

File: backend/agents/synthesizer.py

Input from state:  query, verified_claims, retrieved_content
Output to state:   final_answer, citations, status = "evaluating"

Responsibility:
  Generate the final comprehensive answer using RAG. Retrieve the most
  relevant chunks from ChromaDB, combine with verified claims, and generate
  a cited markdown answer.

Step-by-step flow:

  Step 1 — RAG retrieval
    Query ChromaDB with the original query text
    Retrieve top 15 most semantically similar chunks
    These chunks are the "context" for generation

  Step 2 — Build context block
    Combine: verified_claims + top 15 retrieved chunks
    Format as numbered context items for the LLM

  Step 3 — Generate answer
    System prompt:
    "You are a research synthesis expert. Generate a comprehensive answer to the
    research query using ONLY the provided context. Rules:
    - For every specific fact or statistic, include a citation marker [N]
    - Structure the answer with clear sections using markdown headers
    - Do NOT include any information not present in the context
    - End with a 'Key Takeaways' section with 3-5 bullet points
    - Be comprehensive but not repetitive"

    Pass: system prompt + context block + query
    Output: markdown answer with [1], [2], [3] citation markers

  Step 4 — Extract citations
    Map each [N] marker to the source URL and title it came from
    Build citations list: [{number, url, title, relevant_quote}]

  Step 5 — Update state
    final_answer = generated markdown
    citations = extracted citation list

---

### AGENT 5: EVALUATOR (pipeline step, not a true agent)

File: backend/agents/evaluator.py

Input from state:  query, final_answer, retrieved_content
Output to state:   ragas_scores, status = "done"

Responsibility:
  Run RAGAS evaluation on the final answer and store scores.

Implementation:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

def evaluator_node(state: ResearchState) -> dict:
    # Prepare RAGAS input format
    top_contexts = [c["content"] for c in state["retrieved_content"][:10]]

    data = {
        "question": [state["query"]],
        "answer": [state["final_answer"]],
        "contexts": [top_contexts],
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation (uses Ollama as judge LLM)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ollama_llm,        # same Ollama instance
        embeddings=minilm_embeddings
    )

    scores = {
        "faithfulness": round(result["faithfulness"], 3),
        "answer_relevancy": round(result["answer_relevancy"], 3),
        "context_precision": round(result["context_precision"], 3),
    }
    return {"ragas_scores": scores, "status": "done"}
```

What these scores mean (know this for interviews):
  faithfulness:      Is the answer factually grounded in the retrieved context?
                     Score < 0.5 means the LLM is hallucinating.
  answer_relevancy:  Does the answer actually address the question asked?
                     Score < 0.5 means the answer is off-topic.
  context_precision: Are the retrieved chunks actually relevant to the question?
                     Score < 0.5 means your retrieval is poor — fix embedding or chunking.

---

## 8. TOOLS IMPLEMENTATION

### search_tool

```python
# backend/tools/search.py
from tavily import TavilyClient
import os

def search_web(query: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        search_depth="advanced",   # deeper search, uses more Tavily quota but better results
        max_results=max_results,
        include_raw_content=False  # get snippets first, scrape separately for full content
    )
    return response["results"]
    # Each result: {"url": str, "title": str, "content": str, "score": float}
```

### scrape_tool

```python
# backend/tools/scraper.py
import httpx
from bs4 import BeautifulSoup

async def scrape_page(url: str, timeout: int = 10) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout,
                                    follow_redirects=True,
                                    headers=headers)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "iframe"]):
            tag.decompose()
        # Try to get main content area (most sites use these tags)
        main = (soup.find("article") or
                soup.find("main") or
                soup.find(id="content") or
                soup.find(class_="content") or
                soup.find("body"))
        text = main.get_text(separator=" ", strip=True) if main else ""
        # Cap at 8000 chars to stay within LLM context
        return text[:8000]
    except Exception:
        return ""
```

### vector_store_tool

```python
# backend/tools/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

_model = SentenceTransformer("all-MiniLM-L6-v2")   # load once, reuse
_client = chromadb.PersistentClient(path="./chroma_db")
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def store_content(text: str, metadata: dict, session_id: str) -> int:
    chunks = _splitter.split_text(text)
    if not chunks:
        return 0
    embeddings = _model.encode(chunks).tolist()
    collection = _client.get_or_create_collection(f"session_{session_id}")
    ids = [f"{metadata['url']}_chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))],
        ids=ids
    )
    return len(chunks)

def retrieve_relevant(query: str, session_id: str, k: int = 15) -> list[dict]:
    query_embedding = _model.encode([query]).tolist()[0]
    collection = _client.get_or_create_collection(f"session_{session_id}")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return [
        {"content": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

def delete_session(session_id: str):
    try:
        _client.delete_collection(f"session_{session_id}")
    except Exception:
        pass
```

---

## 9. FASTAPI BACKEND DESIGN

### Schemas

```python
# backend/models/schemas.py
from pydantic import BaseModel
from typing import Optional
import uuid

class ResearchRequest(BaseModel):
    query: str
    max_iterations: int = 3        # default 3 focus areas

class ResearchResponse(BaseModel):
    session_id: str
    status: str

class SessionResult(BaseModel):
    session_id: str
    query: str
    status: str
    focus_areas: list
    retrieved_count: int
    final_answer: Optional[str]
    citations: Optional[list]
    ragas_scores: Optional[dict]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
```

### API Routes

```python
# backend/api/routes.py

POST   /api/research
  Body:    ResearchRequest
  Action:  Create SQLite session, start research graph as background task
  Return:  {session_id, status: "started"}

GET    /api/research/{session_id}
  Return:  SessionResult (full state)

POST   /api/research/{session_id}/chat
  Body:    ChatRequest
  Action:  Retrieve top-k chunks from session ChromaDB, LLM answers using context
  Return:  ChatResponse

GET    /api/sessions
  Return:  List of all SessionResult (summary only, no full answer text)

DELETE /api/research/{session_id}
  Action:  Delete ChromaDB collection + SQLite record

WebSocket /ws/{session_id}
  Events emitted during research:
    {"event": "status_update",     "data": {"status": "researching"}}
    {"event": "focus_area_added",  "data": {"title": "...", "priority": 5}}
    {"event": "content_retrieved", "data": {"url": "...", "chunks_stored": 12}}
    {"event": "claim_verified",    "data": {"claim": "...", "confidence": 0.85}}
    {"event": "answer_ready",      "data": {"answer": "...", "ragas": {...}}}
    {"event": "error",             "data": {"message": "..."}}
```

### Running the graph as a background task

```python
# backend/api/routes.py (key pattern)
from fastapi import BackgroundTasks

@app.post("/api/research")
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    # Save to SQLite
    db_session = create_session(session_id, req.query)
    # Run graph in background — does not block the HTTP response
    background_tasks.add_task(run_research_graph, session_id, req.query, req.max_iterations)
    return {"session_id": session_id, "status": "started"}

async def run_research_graph(session_id: str, query: str, max_iterations: int):
    graph = build_research_graph()
    initial_state = {
        "query": query,
        "session_id": session_id,
        "focus_areas": [],
        "current_focus_index": 0,
        "retrieved_content": [],
        "search_queries_used": [],
        "verified_claims": [],
        "final_answer": "",
        "citations": [],
        "ragas_scores": None,
        "messages": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "errors": [],
        "status": "planning"
    }
    # graph.astream() gives you state after each node — emit WebSocket events here
    async for state_update in graph.astream(initial_state):
        await emit_websocket_event(session_id, state_update)
    # Save final state to SQLite
    update_session_result(session_id, final_state)
```

---

## 10. PROJECT STRUCTURE

```
ai-research-assistant/
│
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── fact_checker.py
│   │   ├── synthesizer.py
│   │   └── evaluator.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py            ← ResearchState TypedDict lives here
│   │   ├── edges.py            ← should_continue() conditional function
│   │   └── research_graph.py  ← StateGraph assembly + compile()
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py           ← Tavily wrapper
│   │   ├── scraper.py          ← httpx + BS4
│   │   └── vector_store.py     ← ChromaDB read/write operations
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py           ← SQLAlchemy ORM models
│   │   └── database.py        ← SQLite engine + session factory
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           ← FastAPI route definitions
│   │   └── websocket.py       ← WebSocket connection manager + emit
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          ← Pydantic request/response schemas
│   │
│   ├── config.py               ← Pydantic Settings, reads from .env
│   └── main.py                 ← FastAPI app creation + route mounting
│
├── frontend/
│   └── app.py                  ← Streamlit UI (all in one file)
│
├── tests/
│   ├── test_tools.py           ← Unit tests for search, scraper, vector_store
│   ├── test_agents.py          ← Unit tests for each agent with mock state
│   └── test_api.py             ← Integration tests for API endpoints
│
├── chroma_db/                  ← Auto-created by ChromaDB PersistentClient
├── research.db                 ← Auto-created SQLite file
│
├── .env.example                ← Template: TAVILY_API_KEY, OLLAMA_BASE_URL, etc.
├── .env                        ← Real secrets (git-ignored)
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md                   ← Must have: setup steps, demo GIF, architecture diagram, RAGAS scores
```

---

## 11. ENVIRONMENT VARIABLES (.env.example)

```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
CHROMA_DB_PATH=./chroma_db
SQLITE_DB_PATH=./research.db
MAX_SCRAPE_TIMEOUT=10
MAX_RESULTS_PER_SEARCH=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=15
LOG_LEVEL=INFO
```

---

## 12. DOCKER COMPOSE

```yaml
version: "3.9"
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./research.db:/app/research.db
    depends_on:
      - ollama

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend

volumes:
  ollama_data:
```

---

## 13. IMPLEMENTATION PHASES (WEEK BY WEEK)

### Week 1 — Foundation + Tools

Day 1-2:
  - Create project structure (all folders and __init__.py files)
  - Set up virtual environment
  - pip install: langgraph langchain langchain-community langchain-ollama
                 chromadb sentence-transformers tavily-python
                 httpx beautifulsoup4 fastapi uvicorn sqlalchemy
                 ragas streamlit python-dotenv pydantic-settings
  - Pull Ollama model: ollama pull llama3.1:8b
  - Write config.py and test Ollama connection (simple "hello" prompt)

Day 3-4:
  - Implement search.py (Tavily) — test with 3 manual queries
  - Implement scraper.py (httpx + BS4) — test on 5 different site types
  - Implement vector_store.py (ChromaDB) — test store + retrieve cycle

Day 5:
  - Write state.py (ResearchState TypedDict — every field with comments)
  - Write tests/test_tools.py
  - Make sure every tool works independently before touching agents

### Week 2 — Agents

Day 1-2:
  - Implement planner.py — test with 5 different queries
    - Get JSON output working reliably before moving on
    - If JSON parsing fails > 30% of the time, improve the prompt

Day 3-4:
  - Implement researcher.py — this is the most complex agent
    - Test each step individually first (generate queries, select URLs, scrape, store)
    - Then wire them together in the node function

Day 5:
  - Implement fact_checker.py
  - Implement synthesizer.py
  - Implement evaluator.py

### Week 3 — Graph + Backend

Day 1-2:
  - Write edges.py (should_continue function)
  - Write research_graph.py (assemble StateGraph, compile)
  - END-TO-END TEST: run a full research query through the graph
    - This is the most important milestone. Don't move to FastAPI until graph works.
    - Test with: "What are the main causes of climate change?"

Day 3-4:
  - Write SQLAlchemy models (db/models.py)
  - Write FastAPI routes (api/routes.py)
  - Write WebSocket handler (api/websocket.py)

Day 5:
  - Test all API endpoints with curl or Postman
  - Fix any async issues (common: blocking calls inside async functions)

### Week 4 — Evaluation + Frontend

Day 1-2:
  - Run RAGAS evaluation on 10 test queries
  - Record scores in a table in your README
  - If faithfulness < 0.6, iterate on synthesizer prompt
  - If context_precision < 0.6, iterate on chunk size or retrieval k

Day 3-5:
  - Build Streamlit app.py:
    - Query input + Submit button
    - Real-time progress: focus areas appearing, content counter updating
    - Final answer displayed as markdown (st.markdown)
    - Citations shown as numbered list with clickable links
    - RAGAS scores shown as metrics (st.metric)
    - Follow-up Q&A input at bottom
    - Sidebar: past sessions list

### Week 5 — Polish + Deploy

Day 1-2:
  - Write Dockerfile and docker-compose.yml
  - Test full stack with Docker Compose
  - Fix any container networking issues

Day 3-4:
  - Write README.md:
    - 1-paragraph project description
    - Architecture diagram (can reuse the SVG from earlier conversation)
    - Setup instructions (clone, copy .env.example, docker-compose up)
    - Screenshot or GIF of UI
    - RAGAS evaluation results table
    - Tech stack table with justifications

Day 5:
  - Deploy backend to Railway or Render (free tier)
  - Deploy Streamlit to Streamlit Community Cloud (free)
  - Add deployed link to README

---

## 14. CRITICAL MISTAKES TO AVOID

1. Do NOT start FastAPI before the LangGraph graph runs end-to-end.
   The graph is the core. FastAPI is just a wrapper around it.

2. Do NOT skip error handling in scraper.py.
   At least 20-30% of URLs will fail (403, timeout, bot detection).
   Every scrape call must be inside try/except returning "" on failure.

3. Do NOT use synchronous requests inside async FastAPI routes.
   Use httpx.AsyncClient, not requests.get. This is a common bug that causes
   FastAPI to freeze under load.

4. Do NOT forget operator.add on retrieved_content in ResearchState.
   Without it, each researcher node call overwrites the list, losing all
   previously retrieved content.

5. Do NOT mix ChromaDB collection namespaces.
   Each research session must use its own collection (f"session_{session_id}").
   Mixing them means retrieval pulls from unrelated research sessions.

6. Do NOT try to evaluate RAGAS without the ragas library version locked.
   RAGAS has breaking changes between minor versions. Pin it in requirements.txt.
   Tested working version: ragas==0.1.x

---

## 15. CV BULLET POINTS (WRITE THESE AFTER BUILDING, USE YOUR ACTUAL NUMBERS)

"Built a multi-agent AI research assistant using LangGraph with 4 specialized agents
(Planner, Researcher, Fact-checker, Synthesizer), autonomously processing 50+ web
sources per research session"

"Implemented RAG pipeline with ChromaDB and MiniLM embeddings, achieving [X] faithfulness
and [Y] answer relevancy on RAGAS evaluation across 10 benchmark queries"

"Deployed with FastAPI (REST + WebSocket real-time streaming) and Streamlit frontend,
fully containerized with Docker Compose"

"Used Ollama + Llama 3.1 8B for fully local LLM inference with zero API cost,
demonstrating understanding of local model deployment"

---

## 16. INTERVIEW TALKING POINTS (PREPARE ANSWERS TO THESE)

Q: Why LangGraph over CrewAI or plain LangChain?
A: Research requires cycles — after fact-checking one focus area, I loop back to
   research the next. LangGraph's StateGraph natively supports this with conditional
   edges. LCEL chains are linear and cannot model this flow.

Q: What does operator.add do in the state TypedDict?
A: It tells LangGraph how to merge state updates from nodes that run in parallel or
   across iterations. Without it, each node overwrites the list. With operator.add,
   each node appends to it. Critical for accumulating research content across the
   Researcher→Fact-checker→Researcher loop.

Q: What does faithfulness measure in RAGAS?
A: It measures whether every claim in the generated answer is actually supported by
   the retrieved context. A faithfulness of 0.85 means 85% of the answer's claims
   are grounded in the source material. Low faithfulness = LLM is hallucinating.

Q: How did you handle LLM calls that return malformed JSON?
A: Three-layer defense: (1) system prompt explicitly says "return ONLY valid JSON
   with no additional text", (2) strip markdown code fences before parsing,
   (3) try/except with a retry on the second attempt with a stricter prompt.

Q: Why did you chunk at 512 tokens with 50 token overlap?
A: 512 tokens is large enough to preserve context within a paragraph but small
   enough that retrieved chunks stay focused. 50 token overlap prevents losing
   information that spans a chunk boundary. These are standard values — I tested
   smaller (256) and larger (1024) chunk sizes and found 512 gave the best
   context_precision score on RAGAS.
```
