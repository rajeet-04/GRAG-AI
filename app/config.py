"""Configuration management using pydantic-settings."""

from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen2.5:7b-q4_k_m", alias="OLLAMA_MODEL")

    # Embedding Model
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")

    # ChromaDB Configuration
    chromadb_path: Path = Field(default=Path("./data/chromadb"), alias="CHROMADB_PATH")

    # Application Settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # OpenAI API Configuration (for OpenWebUI integration)
    api_key: str = Field(default="", alias="API_KEY")
    model_name: str = Field(default="grag-pipeline-v1", alias="MODEL_NAME")

    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Optional: Cloud LLM for Context Builder Agent
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chromadb_path = Path(self.chromadb_path)

    def get_neo4j_uri_with_credentials(self) -> str:
        """Get Neo4j URI with credentials for driver connection."""
        return f"bolt://{self.neo4j_user}:{self.neo4j_password}@{self.neo4j_uri.replace('bolt://', '')}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


class ContextConfig(BaseSettings):
    """Token budget and truncation settings for context management.

    Mirrors RetrievalConfig pattern from Phase 7 for consistency.
    Supports .env override for future LLM upgrades.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    default_token_budget: int = Field(default=8192, alias="CONTEXT_TOKEN_BUDGET")
    warning_threshold_pct: float = Field(
        default=0.80, alias="CONTEXT_WARNING_THRESHOLD"
    )
    tiktoken_model: str = Field(default="cl100k_base", alias="CONTEXT_TIKTOKEN_MODEL")


@lru_cache(maxsize=1)
def get_context_config() -> ContextConfig:
    """Get cached context config instance."""
    return ContextConfig()
