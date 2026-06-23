"""FastAPI application: REST endpoints + a WebSocket that streams live progress.

The LangGraph pipeline is synchronous and CPU/IO heavy, so every run executes in a
worker thread (asyncio.to_thread) to keep the event loop responsive. Progress events
are handed from that thread back to the event loop via loop.call_soon_threadsafe so
the WebSocket can forward them to the browser in real time.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.api.schemas import ResearchRequest, RunDetail, RunSummary
from backend.db import crud
from backend.db.database import SessionLocal, get_session, init_db
from backend.graph.runner import run_streaming

_DONE = object()  # sentinel pushed onto the queue when a run finishes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Research Assistant", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/research", response_model=RunDetail)
async def research(req: ResearchRequest, db: Session = Depends(get_session)):
    """Run a research query to completion (blocking) and return the full result."""
    session_id = uuid.uuid4().hex[:12]
    crud.create_run(db, session_id, req.query)
    state = await asyncio.to_thread(
        run_streaming, req.query, session_id, req.max_iterations
    )
    run = crud.save_result(db, session_id, state)
    return RunDetail(**run.to_dict())


@app.get("/api/history", response_model=list[RunSummary])
def history(db: Session = Depends(get_session)):
    return [RunSummary(**r.to_dict()) for r in crud.list_runs(db)]


@app.get("/api/research/{session_id}", response_model=RunDetail)
def get_result(session_id: str, db: Session = Depends(get_session)):
    run = crud.get_run(db, session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="session not found")
    return RunDetail(**run.to_dict())


@app.websocket("/ws/research")
async def research_ws(websocket: WebSocket):
    """Stream a research run: client sends {query, max_iterations}; server streams
    {type: progress|complete|error} events until completion."""
    await websocket.accept()
    try:
        params = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    query = (params or {}).get("query", "").strip()
    max_iterations = int((params or {}).get("max_iterations", 3))
    if len(query) < 3:
        await websocket.send_json({"type": "error", "message": "query too short"})
        await websocket.close()
        return

    session_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # called from the worker thread -> hop back onto the event loop thread-safely
    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def worker() -> None:
        db = SessionLocal()
        try:
            crud.create_run(db, session_id, query)
            await websocket.send_json({"type": "started", "session_id": session_id})
            state = await asyncio.to_thread(
                run_streaming, query, session_id, max_iterations, emit
            )
            run = crud.save_result(db, session_id, state)
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "complete", "result": run.to_dict()}
            )
        except Exception as e:  # noqa: BLE001 - report failure to the client
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(e)}
            )
        finally:
            db.close()
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    task = asyncio.create_task(worker())
    try:
        while True:
            event = await queue.get()
            if event is _DONE:
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed / disconnected
