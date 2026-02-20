"""Configuration management for Mimic Master."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        # Pinecone Configuration
        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index: str = os.getenv("PINECONE_INDEX", "")

        # Embedding Service Configuration
        self.embedding_provider_url: str = os.getenv(
            "EMBEDDING_PROVIDER_URL", "http://localhost:8000/embed"
        )
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

        # Reranker Service Configuration
        self.reranker_provider_url: str = os.getenv(
            "RERANKER_PROVIDER_URL", "http://localhost:8000/rerank"
        )

        # LangSmith Configuration
        self.langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
        self.langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "mimic-master")
        self.langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.langsmith_endpoint: str = os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )

        # Application Configuration
        self.base_dir: Path = Path(__file__).parent.parent.resolve()

    @property
    def use_mock_embedding(self) -> bool:
        """Check if we should use mock embedding service."""
        return self.embedding_provider_url == "mock"

    @property
    def use_mock_reranker(self) -> bool:
        """Check if we should use mock reranker service."""
        return self.reranker_provider_url == "mock"

    @property
    def is_pinecone_configured(self) -> bool:
        """Check if Pinecone is properly configured."""
        return bool(self.pinecone_api_key and self.pinecone_index)

    @property
    def is_langsmith_configured(self) -> bool:
        """Check if LangSmith is properly configured."""
        return bool(self.langsmith_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


# Global settings instance
settings = get_settings()
