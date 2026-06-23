"""Central configuration, loaded from environment / .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider selection ---
    LLM_PROVIDER: str = "ollama"  # "ollama" | "groq"

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # --- Groq ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    # Judge model for RAGAS. Defaults to the higher daily-token-limit 8b-instant so
    # evaluation doesn't exhaust the free-tier quota of the 70b model. Set to
    # GROQ_MODEL for a stronger judge when you have quota headroom.
    GROQ_JUDGE_MODEL: str = "llama-3.1-8b-instant"

    # --- OpenRouter (preferred RAGAS judge: reliable structured output, pay-per-use) ---
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_JUDGE_MODEL: str = "openai/gpt-4o-mini"

    # --- Web search ---
    TAVILY_API_KEY: str = ""

    # --- Observability (LangSmith tracing, opt-in) ---
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "ai-research-assistant"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # --- Storage ---
    CHROMA_DB_PATH: str = "./chroma_db"
    SQLITE_DB_PATH: str = "./research.db"

    # --- Tunables ---
    MAX_SCRAPE_TIMEOUT: int = 10
    MAX_RESULTS_PER_SEARCH: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 8
    # Cosine-distance ceiling for a chunk to count as relevant context. Lower =
    # stricter retrieval = higher RAGAS context precision. Floored by MIN_CONTEXT_CHUNKS
    # so the synthesizer is never starved when every chunk is borderline.
    CONTEXT_MAX_DISTANCE: float = 0.70
    MIN_CONTEXT_CHUNKS: int = 5
    LOG_LEVEL: str = "INFO"


settings = Settings()
