"""Smoke test for the embedding service configured by the current environment."""

import os
import importlib

import pytest


@pytest.mark.asyncio
async def test_embedding_current_settings() -> None:
    """Validate the current embedding configuration and call the provider once."""
    if os.getenv("RUN_REAL_EMBEDDING") != "1":
        pytest.skip("Set RUN_REAL_EMBEDDING=1 to validate the current embedding setup.")

    from mimic_master import config
    from mimic_master.services import embedding_service

    importlib.reload(config)
    importlib.reload(embedding_service)

    settings = config.Settings()
    assert settings.embedding_backend in {"openrouter", "nvidia", "local"}
    assert settings.embedding_api_key
    assert settings.embedding_base_url
    assert settings.embedding_model

    response = await embedding_service.EmbeddingService().embed(
        ["embedding configuration smoke test"]
    )

    assert len(response.embeddings) == 1
    assert len(response.embeddings[0].dense) == settings.embedding_dimension
