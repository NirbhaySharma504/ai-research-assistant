"""Provider-agnostic LLM factory.

Every agent calls get_llm() instead of instantiating a provider directly, so the
whole pipeline can switch between local Ollama (llama3.2:3b) and hosted Groq
(llama-3.3-70b) with a single env var (LLM_PROVIDER). The RAGAS judge prefers Groq
for reliable structured output and falls back to Ollama when no Groq key is set.
"""

from functools import lru_cache

from backend.config import settings


def get_llm(temperature: float = 0.3, provider: str | None = None, model: str | None = None):
    """Return a LangChain chat model for the configured (or overridden) provider."""
    provider = (provider or settings.LLM_PROVIDER).lower()

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or settings.GROQ_MODEL,
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            max_retries=8,  # ride out free-tier 429s (used heavily as the RAGAS judge)
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        temperature=temperature,
        base_url=settings.OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_judge_llm():
    """LLM used by RAGAS as the evaluation judge.

    Reliable structured output matters here: a weak judge makes RAGAS return NaN
    (notably for context precision). Preference order:
      1. OpenRouter (pay-per-use, gpt-4o-mini) — most reliable, used for benchmarking
      2. Groq (free tier) — fast but the small models NaN context precision
      3. local Ollama — always available, noisier
    """
    if settings.OPENROUTER_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.OPENROUTER_JUDGE_MODEL,
            temperature=0.0,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            max_retries=5,
        )
    if settings.GROQ_API_KEY:
        return get_llm(
            temperature=0.0, provider="groq", model=settings.GROQ_JUDGE_MODEL
        )
    return get_llm(temperature=0.0, provider="ollama")
