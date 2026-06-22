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

    # --- Web search ---
    TAVILY_API_KEY: str = ""

    # --- Storage ---
    CHROMA_DB_PATH: str = "./chroma_db"
    SQLITE_DB_PATH: str = "./research.db"

    # --- Tunables ---
    MAX_SCRAPE_TIMEOUT: int = 10
    MAX_RESULTS_PER_SEARCH: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 15
    LOG_LEVEL: str = "INFO"


settings = Settings()
