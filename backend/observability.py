"""Opt-in LangSmith tracing.

LangGraph/LangChain emit traces automatically when the LANGCHAIN_* env vars are
present, so this just promotes the values from settings (.env) into os.environ once,
before any LLM call. No-ops unless LANGCHAIN_TRACING_V2=true and an API key are set.
"""

import os

from backend.config import settings

_configured = False


def setup_tracing() -> bool:
    """Enable LangSmith tracing if configured. Returns True if tracing is active."""
    global _configured
    if _configured:
        return settings.LANGCHAIN_TRACING_V2 and bool(settings.LANGCHAIN_API_KEY)
    _configured = True

    if not (settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY):
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    return True
