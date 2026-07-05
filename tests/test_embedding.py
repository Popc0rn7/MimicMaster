"""Tests for embedding service."""

import pytest

from mimic_master.config import settings
from mimic_master.services.embedding_service import EmbeddingService
from mimic_master.models.embeddings import DenseAndSparseEmbeddings, SparseVector


@pytest.fixture
def embedding_service(monkeypatch):
    """Create a fresh embedding service with a fake local provider."""
    service = EmbeddingService()
    monkeypatch.setattr(
        "mimic_master.services.embedding_service.settings.embedding_backend",
        "local",
    )

    async def fake_openai_compatible_embed(texts):
        return service._mock_embed(texts)

    monkeypatch.setattr(
        service, "_openai_compatible_embed", fake_openai_compatible_embed
    )
    return service


@pytest.mark.asyncio
async def test_embedding_dispatches_to_nvidia_by_default(
    monkeypatch, embedding_service
):
    """NVIDIA is the default embedding provider."""
    monkeypatch.setattr(
        "mimic_master.services.embedding_service.settings.embedding_backend",
        "nvidia",
    )
    monkeypatch.setattr(
        "mimic_master.services.embedding_service.settings.nvidia_api_key",
        "test-key",
        raising=False,
    )

    async def fake_openai_compatible_embed(texts):
        return embedding_service._mock_embed(texts)

    monkeypatch.setattr(
        embedding_service, "_openai_compatible_embed", fake_openai_compatible_embed
    )

    response = await embedding_service.embed(["Fireball"])

    assert len(response.embeddings) == 1


@pytest.mark.asyncio
async def test_embedding_dispatches_to_local_provider(monkeypatch, embedding_service):
    """Local is the self-hosted OpenAI-compatible embedding provider."""
    monkeypatch.setattr(
        "mimic_master.services.embedding_service.settings.embedding_backend",
        "local",
    )

    async def fake_openai_compatible_embed(texts):
        return embedding_service._mock_embed(texts)

    monkeypatch.setattr(
        embedding_service, "_openai_compatible_embed", fake_openai_compatible_embed
    )

    response = await embedding_service.embed(["Goblin"])

    assert len(response.embeddings) == 1


@pytest.mark.asyncio
async def test_embedding_service(embedding_service):
    """Test basic embedding generation."""
    response = await embedding_service.embed(["Hello, world!"])

    assert len(response.embeddings) == 1
    assert len(response.embeddings[0].dense) > 0
    assert response.dense_dimension > 0


@pytest.mark.asyncio
async def test_embedding_multiple_texts(embedding_service):
    """Test embedding multiple texts."""
    response = await embedding_service.embed(["First text", "Second text"])

    assert len(response.embeddings) == 2
    assert len(response.embeddings[0].dense) > 0
    assert len(response.embeddings[1].dense) > 0


@pytest.mark.asyncio
async def test_sparse_embedding(embedding_service):
    """Test that sparse embeddings are generated for the fake HTTP provider."""
    response = await embedding_service.embed(["fireball spell damage"])

    assert len(response.embeddings) == 1
    emb: DenseAndSparseEmbeddings = response.embeddings[0]

    assert len(emb.dense) > 0
    assert isinstance(emb.sparse, SparseVector)
    if not settings.use_nvidia_embedding:
        assert len(emb.sparse.indices) > 0 or len(emb.sparse.values) > 0


@pytest.mark.asyncio
async def test_embedding_consistency(embedding_service):
    """Test that the fake provider is deterministic."""

    text = "D&D 5E rules"

    response1 = await embedding_service.embed([text])
    response2 = await embedding_service.embed([text])

    assert response1.embeddings[0].dense == response2.embeddings[0].dense


@pytest.mark.asyncio
async def test_openai_compatible_embed_rejects_empty_dense_response(monkeypatch):
    """Provider empty responses should fail before later iteration errors."""
    service = EmbeddingService()

    async def fake_dense_embed(texts):
        return None

    async def fake_sparse_embed(texts):
        return [SparseVector(indices=[], values=[]) for _ in texts]

    monkeypatch.setattr(service, "_dense_embed", fake_dense_embed)
    monkeypatch.setattr(service, "_pinecone_sparse_embed", fake_sparse_embed)
    monkeypatch.setattr(
        "mimic_master.services.embedding_service.settings.embedding_backend",
        "openrouter",
    )

    with pytest.raises(RuntimeError, match="returned no dense embeddings"):
        await service._openai_compatible_embed(["test"])
