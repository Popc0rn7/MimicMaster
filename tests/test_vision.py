"""Tests for vision service."""

import pytest
from pathlib import Path

# Test image path - use Aarakocra (first monster in MM)
TEST_IMAGE_PATH = "img/bestiary/MM/Aarakocra.jpg"


@pytest.fixture
def test_image_path():
    """Get the test image path."""
    base_dir = Path(__file__).parent.parent
    return str(base_dir / "knowledge" / "raw" / TEST_IMAGE_PATH)


@pytest.fixture
def vision_service():
    """Create vision service instance."""
    from mimic_master.services.vision_service import VisionService

    return VisionService()


@pytest.mark.asyncio
async def test_describe_image_mock(vision_service, test_image_path):
    """Test image description in mock mode."""
    # Force mock mode
    from mimic_master import config

    original_url = config.settings.vision_provider_url
    config.settings.vision_provider_url = "mock"

    try:
        result = await vision_service.describe_image(test_image_path)

        assert result is not None
        assert result.image_path == test_image_path
        assert result.description
        assert "Mock" in result.description or "模拟" in result.description
        print(f"\n[Mock Mode] Description: {result.description}")
    finally:
        config.settings.vision_provider_url = original_url


def test_vision_service_initialization():
    """Test vision service can be initialized."""
    from mimic_master.services.vision_service import VisionService
    from mimic_master.config import settings

    service = VisionService()

    assert service._model == settings.vision_model
    assert service._base_url == settings.vision_provider_url


def test_vision_config():
    """Test vision configuration."""
    from mimic_master.config import settings

    assert settings.vision_model
    assert isinstance(settings.use_mock_vision, bool)


def test_chat_completions_url_accepts_openai_compatible_base_url():
    """Vision provider URL can be an OpenAI-compatible /v1 base URL."""
    from mimic_master.services.vision_service import build_chat_completions_url

    assert (
        build_chat_completions_url("https://vision.example/v1")
        == "https://vision.example/v1/chat/completions"
    )


def test_chat_completions_url_accepts_provider_root_url():
    """Vision provider URL can be a provider root that exposes OpenAI /v1."""
    from mimic_master.services.vision_service import build_chat_completions_url

    assert (
        build_chat_completions_url("https://vision.example")
        == "https://vision.example/v1/chat/completions"
    )


def test_chat_completions_url_accepts_full_endpoint():
    """Vision provider URL can be a full chat completions endpoint."""
    from mimic_master.services.vision_service import build_chat_completions_url

    assert (
        build_chat_completions_url("https://vision.example/v1/chat/completions")
        == "https://vision.example/v1/chat/completions"
    )
