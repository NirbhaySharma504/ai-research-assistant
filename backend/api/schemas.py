"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The research question")
    max_iterations: int = Field(3, ge=1, le=6)


class RunSummary(BaseModel):
    session_id: str
    query: str
    status: str
    source_count: int
    ragas_scores: dict | None = None
    created_at: str | None = None


class RunDetail(RunSummary):
    final_answer: str = ""
    focus_areas: list = []
    citations: list = []
    errors: list = []
    iteration_count: int = 0
    completed_at: str | None = None
