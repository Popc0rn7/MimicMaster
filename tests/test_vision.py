"""Tests for vision service."""

import pytest
import os
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


@pytest.mark.asyncio
async def test_describe_image_real(vision_service, test_image_path):
    """Test image description with real GLM-4V API."""
    from mimic_master.config import settings
    if not settings.zhipu_api_key:
        pytest.skip("ZHIPU_API_KEY not set - requires real API key")

    result = await vision_service.describe_image(test_image_path)

    assert result is not None
    assert result.image_path == test_image_path
    assert result.description
    assert len(result.description) > 10  # Should have meaningful description
    print(f"\n[Real API] Description: {result.description}")


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

    # Should have API key configured
    assert settings.zhipu_api_key
    # Should use real API (not mock)
    assert not settings.use_mock_vision
