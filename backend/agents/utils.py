"""Shared agent helpers: robust JSON extraction from LLM output."""

import json
import re
from typing import Any


def parse_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating markdown fences and prose.

    Three-layer defense (matches the interview talking point):
    1. strip ```json / ``` fences
    2. try a direct json.loads
    3. fall back to extracting the first {...} or [...] block
    """
    cleaned = text.strip()

    # 1. strip code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 2. direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. extract first JSON object or array
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}")


def invoke_json(llm, prompt: str, retries: int = 1) -> Any:
    """Invoke the LLM and parse JSON, retrying once with a stricter nudge."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        msg = prompt
        if attempt > 0:
            msg = prompt + "\n\nReturn ONLY valid JSON. No prose, no code fences."
        resp = llm.invoke(msg)
        content = resp.content if hasattr(resp, "content") else str(resp)
        try:
            return parse_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise ValueError(f"LLM did not return valid JSON after retries: {last_err}")
