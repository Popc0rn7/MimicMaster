"""E2E smoke test for the configured vision model."""

from __future__ import annotations

import asyncio
import base64
import os

import httpx
import pytest

from tests.e2e.helpers import require_run_e2e, reload_config

pytestmark = pytest.mark.e2e

_ONE_PIXEL_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QA"
    "FAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/a"
    "AAwDAQACEQMRAD8AVMH/2Q=="
)


@pytest.mark.asyncio
async def test_vision_model_describes_image(tmp_path) -> None:
    """Call the configured vision provider with a tiny local image."""
    require_run_e2e()

    config = reload_config()
    settings = config.Settings()
    if not settings.is_vision_configured:
        pytest.skip(
            "Vision model not configured. Set VISION_API_KEY and VISION_PROVIDER_URL."
        )

    image_path = tmp_path / "one_pixel.jpg"
    image_path.write_bytes(base64.b64decode(_ONE_PIXEL_JPEG))

    from mimic_master.services.vision_service import VisionService
    from mimic_master.services.embedding_service import normalize_proxy_environment

    normalize_proxy_environment()
    timeout_seconds = float(os.getenv("E2E_VISION_TIMEOUT_SECONDS", "20"))
    try:
        result = await asyncio.wait_for(
            VisionService().describe_image(
                str(image_path),
                prompt="请用一句中文简短描述这张测试图片。",
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        pytest.fail(
            "Vision provider request timed out: "
            f"timeout={timeout_seconds:g}s "
            f"model={settings.vision_model!r} "
            f"url={settings.vision_provider_url!r}"
        )
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text[:500]
        pytest.fail(
            "Vision provider request failed: "
            f"status={exc.response.status_code} "
            f"model={settings.vision_model!r} "
            f"url={settings.vision_provider_url!r} "
            f"body={response_text!r}"
        )

    assert result.image_path == str(image_path)
    assert result.description.strip()
