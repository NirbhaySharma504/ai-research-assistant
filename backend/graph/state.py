"""Shared LangGraph state. Every agent reads from and writes to ResearchState."""

import operator
from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict  # pydantic requires this on Python < 3.12


class FocusArea(TypedDict):
    title: str            # "Current research landscape in X"
    description: str      # what to look for (2-3 sentences)
    priority: int         # 1-5, highest priority tackled first
    status: str           # "pending" | "in_progress" | "completed"


class RetrievedContent(TypedDict):
    url: str
    title: str
    content: str          # cleaned text, capped at 8000 chars
    focus_area: str       # which focus area this was retrieved for
    timestamp: str
    relevance_score: float


class VerifiedClaim(TypedDict):
    claim: str
    is_verified: bool
    confidence: float                 # 0.0 - 1.0
    supporting_sources: List[str]
    contradicting_sources: List[str]


class ResearchState(TypedDict):
    # --- INPUT ---
    query: str
    session_id: str

    # --- PLANNER OUTPUT ---
    focus_areas: List[FocusArea]
    current_focus_index: int

    # --- RESEARCHER OUTPUT ---
    # operator.add => nodes append to (not overwrite) these lists across iterations.
    retrieved_content: Annotated[List[RetrievedContent], operator.add]
    search_queries_used: List[str]

    # --- FACT-CHECKER OUTPUT ---
    verified_claims: List[VerifiedClaim]

    # --- SYNTHESIZER OUTPUT ---
    final_answer: str                 # markdown
    citations: List[dict]             # [{number, url, title, quote}]

    # --- EVALUATION ---
    ragas_scores: Optional[dict]

    # --- CONTROL ---
    messages: Annotated[List[BaseMessage], operator.add]
    iteration_count: int
    max_iterations: int
    errors: List[str]
    status: str  # planning|researching|fact_checking|synthesizing|evaluating|done|error
