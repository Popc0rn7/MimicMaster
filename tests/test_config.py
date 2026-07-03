"""Tests for application configuration."""

from mimic_master.config import Settings


def test_embedding_defaults_to_nvidia(monkeypatch):
    """Embedding defaults to NVIDIA BGE-M3."""
    for key in (
        "EMBEDDING_PROVIDER_TYPE",
        "EMBEDDING_PROVIDER_URL",
        "PROVIDER_BASE_URL",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.embedding_provider_type == "nvidia"
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.nvidia_embedding_model == "baai/bge-m3"


def test_nvidia_api_key_preferred_over_legacy_openai_key(monkeypatch):
    """NVIDIA_API_KEY is preferred while OPENAI_API_KEY remains a fallback."""
    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    settings = Settings()

    assert settings.nvidia_api_key == "nvidia-key"
    assert settings.use_nvidia_embedding


def test_legacy_openai_key_still_enables_nvidia_provider(monkeypatch):
    """OPENAI_API_KEY remains a compatibility fallback for NVIDIA auth."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "nvidia")

    settings = Settings()

    assert settings.nvidia_api_key == "legacy-key"
    assert settings.use_nvidia_embedding


def test_http_provider_uses_local_endpoint(monkeypatch):
    """HTTP provider uses the configured endpoint as the local/self-hosted option."""
    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "http")
    monkeypatch.setenv("EMBEDDING_PROVIDER_URL", "http://localhost:8001/embeddings")

    settings = Settings()

    assert settings.use_http_embedding
    assert settings.embedding_provider_url == "http://localhost:8001/embeddings"


def test_mongodb_uri_uses_admin_auth_source_with_credentials(monkeypatch):
    """Docker Mongo root credentials authenticate against the admin database."""
    monkeypatch.setenv("MONGODB_HOST", "localhost")
    monkeypatch.setenv("MONGODB_PORT", "27017")
    monkeypatch.setenv("MONGODB_USERNAME", "mimic")
    monkeypatch.setenv("MONGODB_PASSWORD", "mimic_master")

    settings = Settings()

    assert (
        settings.mongodb_uri
        == "mongodb://mimic:mimic_master@localhost:27017/?authSource=admin"
    )
