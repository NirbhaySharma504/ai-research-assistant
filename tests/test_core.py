"""Fast, offline unit tests for pure logic (no network / LLM / DB)."""

import pytest

from backend.agents.evaluator import _clean
from backend.agents.utils import parse_json
from backend.graph.edges import should_continue


# --------------------------------------------------------------- parse_json
def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fence():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_extracts_from_prose():
    text = 'Sure! Here you go: {"focus_areas": [1, 2]} hope that helps'
    assert parse_json(text) == {"focus_areas": [1, 2]}


def test_parse_json_array():
    assert parse_json('["a", "b"]') == ["a", "b"]


def test_parse_json_raises_on_garbage():
    with pytest.raises((ValueError, Exception)):
        parse_json("no json here at all")


# --------------------------------------------------------------- _clean (RAGAS)
def test_clean_valid_number_is_rounded():
    assert _clean(0.87654) == 0.877


def test_clean_nan_becomes_none():
    assert _clean(float("nan")) is None


def test_clean_non_numeric_becomes_none():
    assert _clean("oops") is None


# --------------------------------------------------------------- should_continue
def _state(idx, n_focus, iters=0, max_iters=3, status="researching"):
    return {
        "current_focus_index": idx,
        "focus_areas": [{} for _ in range(n_focus)],
        "iteration_count": iters,
        "max_iterations": max_iters,
        "status": status,
    }


def test_continue_when_focus_areas_remain():
    assert should_continue(_state(idx=1, n_focus=3)) == "continue_research"


def test_synthesize_when_all_focus_done():
    assert should_continue(_state(idx=3, n_focus=3)) == "synthesize"


def test_synthesize_when_over_iteration_limit():
    assert should_continue(_state(idx=1, n_focus=5, iters=3, max_iters=3)) == "synthesize"


def test_synthesize_on_error_status():
    assert should_continue(_state(idx=0, n_focus=5, status="error")) == "synthesize"
