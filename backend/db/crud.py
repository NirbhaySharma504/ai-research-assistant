"""Small data-access helpers for ResearchRun rows."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import ResearchRun


def create_run(db: Session, session_id: str, query: str) -> ResearchRun:
    run = ResearchRun(session_id=session_id, query=query, status="running")
    db.add(run)
    db.commit()
    return run


def save_result(db: Session, session_id: str, state: dict) -> ResearchRun:
    """Persist the final ResearchState onto the run row."""
    run = db.get(ResearchRun, session_id)
    if run is None:
        run = ResearchRun(session_id=session_id, query=state.get("query", ""))
        db.add(run)

    run.final_answer = state.get("final_answer", "")
    run.focus_areas = state.get("focus_areas", [])
    run.citations = state.get("citations", [])
    run.ragas_scores = state.get("ragas_scores")
    run.errors = state.get("errors", [])
    run.iteration_count = state.get("iteration_count", 0)
    run.source_count = len(state.get("retrieved_content", []))
    run.status = state.get("status", "done")
    run.completed_at = datetime.now(timezone.utc)
    db.commit()
    return run


def get_run(db: Session, session_id: str) -> ResearchRun | None:
    return db.get(ResearchRun, session_id)


def list_runs(db: Session, limit: int = 50) -> list[ResearchRun]:
    stmt = select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))
