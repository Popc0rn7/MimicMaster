"""Tests for application configuration."""

from mimic_master.config import Settings


def test_embedding_defaults_to_nvidia(monkeypatch):
    """Embedding defaults to NVIDIA BGE-M3."""
    for key in (
        "EMBEDDING_BACKEND",
        "PROVIDER_BASE_URL",
        "EMBEDDING_MODEL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "LOCAL_API_KEY",
        "LOCAL_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.embedding_backend == "nvidia"
    assert settings.embedding_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.embedding_model == "baai/bge-m3"


def test_nvidia_backend_uses_nvidia_credentials(monkeypatch):
    """NVIDIA backend uses NVIDIA API settings."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "nvidia")
    monkeypatch.setenv("EMBEDDING_MODEL", "shared-model")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.example/v1")

    settings = Settings()

    assert settings.embedding_api_key == "nvidia-key"
    assert settings.embedding_base_url == "https://nvidia.example/v1"
    assert settings.embedding_model == "shared-model"
    assert settings.use_nvidia_embedding


def test_openrouter_backend_uses_openrouter_credentials(monkeypatch):
    """OpenRouter backend uses OpenRouter API settings."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openrouter")
    monkeypatch.setenv("EMBEDDING_MODEL", "shared-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/api/v1")

    settings = Settings()

    assert settings.embedding_api_key == "openrouter-key"
    assert settings.embedding_base_url == "https://openrouter.example/api/v1"
    assert settings.embedding_model == "shared-model"
    assert settings.use_openrouter_embedding


def test_local_backend_uses_local_openai_compatible_settings(monkeypatch):
    """Local backend uses configured OpenAI-compatible local settings."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "shared-model")
    monkeypatch.setenv("LOCAL_API_KEY", "local-key")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8001/v1")

    settings = Settings()

    assert settings.embedding_api_key == "local-key"
    assert settings.embedding_base_url == "http://localhost:8001/v1"
    assert settings.embedding_model == "shared-model"
    assert settings.use_local_embedding


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


def test_vision_uses_provider_neutral_api_key(monkeypatch):
    """Vision accepts provider-neutral OpenAI-compatible credentials."""
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("VISION_API_KEY", "vision-key")
    monkeypatch.setenv("VISION_PROVIDER_URL", "https://vision.example/v1")
    monkeypatch.setenv("VISION_MODEL", "gpt-5.4")

    settings = Settings()

    assert settings.vision_api_key == "vision-key"
    assert settings.zhipu_api_key == "vision-key"
    assert settings.vision_provider_url == "https://vision.example/v1"
    assert settings.vision_model == "gpt-5.4"
    assert settings.is_vision_configured
    assert not settings.use_mock_vision


def test_vision_keeps_zhipu_api_key_as_compatibility_fallback(monkeypatch):
    """Vision still accepts the previous ZHIPU_API_KEY environment variable."""
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-key")
    monkeypatch.setenv("VISION_PROVIDER_URL", "https://vision.example/v1")

    settings = Settings()

    assert settings.vision_api_key == "zhipu-key"
    assert settings.zhipu_api_key == "zhipu-key"
    assert settings.is_vision_configured


def test_empty_vision_api_key_falls_back_to_zhipu_key(monkeypatch):
    """An empty provider-neutral key does not mask the legacy key."""
    monkeypatch.setenv("VISION_API_KEY", "")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-key")
    monkeypatch.setenv("VISION_PROVIDER_URL", "https://vision.example/v1")

    settings = Settings()

    assert settings.vision_api_key == "zhipu-key"
    assert settings.is_vision_configured
