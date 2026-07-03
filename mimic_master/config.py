"""Configuration management for Mimic Master."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        # Pinecone Configuration
        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index: str = os.getenv("PINECONE_INDEX", "mimic-rules-prod")

        # Provider Base URL for self-hosted HTTP services
        self.provider_base_url: str = os.getenv("PROVIDER_BASE_URL", "")

        # Embedding Service Configuration
        embed_url = os.getenv("EMBEDDING_PROVIDER_URL", "")
        if not embed_url and self.provider_base_url:
            embed_url = f"{self.provider_base_url}/embeddings"
        self.embedding_provider_url: str = embed_url
        self.embedding_provider_type: str = os.getenv(
            "EMBEDDING_PROVIDER_TYPE", "nvidia"
        ).lower()
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        self.nvidia_api_key: str = os.getenv("NVIDIA_API_KEY") or os.getenv(
            "OPENAI_API_KEY", ""
        )
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.nvidia_base_url: str = os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.nvidia_embedding_model: str = os.getenv(
            "NVIDIA_EMBEDDING_MODEL", "baai/bge-m3"
        )

        # Reranker Service Configuration (uses Pinecone native inference API)
        rerank_url = os.getenv("RERANKER_PROVIDER_URL", "")
        if not rerank_url and self.provider_base_url:
            rerank_url = f"{self.provider_base_url}/reranker"
        self.reranker_provider_url: str = rerank_url or "mock"
        self.reranker_provider_type: str = os.getenv(
            "RERANKER_PROVIDER_TYPE", "pinecone"
        ).lower()

        # Vision Service Configuration
        vision_url = os.getenv("VISION_PROVIDER_URL", "")
        if not vision_url and self.provider_base_url:
            vision_url = f"{self.provider_base_url}/vision"
        self.vision_provider_url: str = vision_url or "mock"
        self.zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
        self.vision_model: str = os.getenv("VISION_MODEL", "glm-4v")

        # LangSmith Configuration
        self.langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
        self.langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "mimic-master")
        self.langsmith_tracing: bool = (
            os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        )
        self.langsmith_endpoint: str = os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )

        # Namespace Configuration (from NAMING.md)
        self.rules_namespace: str = os.getenv("RULES_NAMESPACE", "rules")
        self.episodes_namespace: str = os.getenv("EPISODES_NAMESPACE", "episodes")
        self.monsters_namespace: str = os.getenv("MONSTERS_NAMESPACE", "monsters")
        self.spells_namespace: str = os.getenv("SPELLS_NAMESPACE", "spells")

        # MongoDB Configuration
        self.mongodb_host: str = os.getenv("MONGODB_HOST", "localhost")
        self.mongodb_port: int = int(os.getenv("MONGODB_PORT", "27017"))
        self.mongodb_database: str = os.getenv("MONGODB_DATABASE", "mimic_master")
        self.mongodb_username: str = os.getenv("MONGODB_USERNAME", "")
        self.mongodb_password: str = os.getenv("MONGODB_PASSWORD", "")

        # Application Configuration
        self.base_dir: Path = Path(__file__).parent.parent.resolve()
        self.project_name: str = "mimic-master"

    @property
    def use_mock_embedding(self) -> bool:
        """Deprecated: runtime mock embedding provider is no longer supported."""
        return False

    @property
    def use_http_embedding(self) -> bool:
        """Check if we should use a self-hosted HTTP embedding service."""
        return self.embedding_provider_type == "http"

    @property
    def use_mock_reranker(self) -> bool:
        """Check if we should use mock reranker service."""
        return self.reranker_provider_type == "mock"

    @property
    def use_pinecone_reranker(self) -> bool:
        """Check if we should use Pinecone native reranker."""
        return self.reranker_provider_type == "pinecone"

    @property
    def use_nvidia_embedding(self) -> bool:
        """Check if we should use NVIDIA API for embedding."""
        return self.embedding_provider_type == "nvidia" and bool(self.nvidia_api_key)

    @property
    def use_mock_vision(self) -> bool:
        """Check if we should use mock vision service."""
        return self.vision_provider_url == "mock" or not self.zhipu_api_key

    @property
    def is_vision_configured(self) -> bool:
        """Check if vision service is properly configured."""
        return bool(self.zhipu_api_key and self.vision_provider_url != "mock")

    @property
    def is_pinecone_configured(self) -> bool:
        """Check if Pinecone is properly configured."""
        return bool(self.pinecone_api_key and self.pinecone_index)

    @property
    def is_langsmith_configured(self) -> bool:
        """Check if LangSmith is properly configured."""
        return bool(self.langsmith_api_key)

    @property
    def mongodb_uri(self) -> str:
        """Build MongoDB connection URI."""
        if self.mongodb_username and self.mongodb_password:
            return f"mongodb://{self.mongodb_username}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}"
        return f"mongodb://{self.mongodb_host}:{self.mongodb_port}"

    @property
    def is_mongodb_configured(self) -> bool:
        """Check if MongoDB is properly configured."""
        return bool(self.mongodb_host)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


# Global settings instance
settings = get_settings()
