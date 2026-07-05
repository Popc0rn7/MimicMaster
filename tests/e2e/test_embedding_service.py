"""E2E smoke test for the configured embedding service."""

from __future__ import annotations

import importlib
import math

import pytest

from tests.e2e.helpers import require_run_e2e, reload_config

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_embedding_service_current_config() -> None:
    """Prove the selected embedding backend returns finite dense vectors."""
    require_run_e2e()

    config = reload_config()
    from mimic_master.services import embedding_service

    importlib.reload(embedding_service)

    settings = config.Settings()
    assert settings.is_embedding_configured, (
        "Embedding backend is not configured. Set EMBEDDING_BACKEND, API key, "
        "base URL, and EMBEDDING_MODEL."
    )

    response = await embedding_service.EmbeddingService().embed(
        ["D&D 5E E2E smoke test: Fireball deals fire damage."]
    )

    assert len(response.embeddings) == 1
    dense = response.embeddings[0].dense
    assert len(dense) == settings.embedding_dimension
    assert all(isinstance(value, float) and math.isfinite(value) for value in dense)
