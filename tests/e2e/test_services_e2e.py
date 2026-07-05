"""End-to-end smoke tests for real service availability.

These tests are intentionally skipped by default. They call configured external
services and should be run when validating deployment/service wiring:

    RUN_SERVICE_E2E=1 uv run pytest tests/e2e/test_services_e2e.py -v
"""

from __future__ import annotations

import importlib
import math
import os

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _service_e2e_enabled() -> bool:
    return os.getenv("RUN_SERVICE_E2E") == "1"


def _reload_configured_modules():
    from mimic_master import config
    from mimic_master.services import embedding_service

    importlib.reload(config)
    importlib.reload(embedding_service)
    return config, embedding_service


@pytest.mark.asyncio
async def test_embedding_service_end_to_end_current_config() -> None:
    """A single test that proves the configured embedding backend can embed text."""
    if not (_service_e2e_enabled() or os.getenv("RUN_REAL_EMBEDDING") == "1"):
        pytest.skip(
            "Set RUN_SERVICE_E2E=1 or RUN_REAL_EMBEDDING=1 to call embedding service."
        )

    config, embedding_service = _reload_configured_modules()
    settings = config.Settings()
    assert settings.is_embedding_configured, (
        "Embedding backend is not configured. Set EMBEDDING_BACKEND, API key, "
        "base URL, and EMBEDDING_MODEL."
    )

    service = embedding_service.EmbeddingService()
    response = await service.embed(
        ["D&D 5E service smoke test: Fireball deals fire damage."]
    )

    assert len(response.embeddings) == 1
    dense = response.embeddings[0].dense
    assert len(dense) == settings.embedding_dimension
    assert all(isinstance(value, float) and math.isfinite(value) for value in dense)


@pytest.mark.asyncio
async def test_agent_api_end_to_end_current_app() -> None:
    """A single test that proves the configured Agent API can process a request."""
    if not (_service_e2e_enabled() or os.getenv("RUN_AGENT_E2E") == "1"):
        pytest.skip("Set RUN_SERVICE_E2E=1 or RUN_AGENT_E2E=1 to call Agent API.")

    from mimic_master.api.app import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/agent/with-context",
            json={
                "query": "用一句话描述酒馆里玩家看到的场景。",
                "session_id": "e2e-agent-smoke",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["session_id"] == "e2e-agent-smoke"
    assert isinstance(payload["response"], str)
    assert payload["response"].strip()
    assert isinstance(payload["retrieved_context"], list)

    if os.getenv("REQUIRE_REAL_AGENT_LLM") == "1":
        assert "full implementation" not in payload["response"].lower()
