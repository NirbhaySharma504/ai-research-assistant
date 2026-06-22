"""Planner agent: decomposes a query into 3-5 prioritized research focus areas."""

from backend.agents.utils import invoke_json
from backend.graph.state import ResearchState
from backend.llm import get_llm

SYSTEM_PROMPT = """You are a research planning expert. Given a user research query, \
generate exactly 3 to 5 research focus areas that together would comprehensively \
answer the query.

Rules:
- Each focus area must be specific and non-overlapping with others
- Cover different dimensions: background/history, current state, technical details, \
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
}"""


def planner_node(state: ResearchState) -> dict:
    llm = get_llm(temperature=0.3)
    prompt = f"{SYSTEM_PROMPT}\n\nResearch query: {state['query']}"

    try:
        data = invoke_json(llm, prompt)
        raw = data["focus_areas"]
    except Exception as e:  # noqa: BLE001
        return {
            "errors": state.get("errors", []) + [f"planner: {e}"],
            "status": "error",
        }

    focus_areas = [
        {
            "title": fa.get("title", "Untitled"),
            "description": fa.get("description", ""),
            "priority": int(fa.get("priority", 3)),
            "status": "pending",
        }
        for fa in raw
    ]
    focus_areas.sort(key=lambda f: f["priority"], reverse=True)

    return {
        "focus_areas": focus_areas,
        "current_focus_index": 0,
        "status": "researching",
    }
