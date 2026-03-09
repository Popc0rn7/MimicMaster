"""Tests for embedding service."""

import pytest

from mimic_master.services.embedding_service import EmbeddingService
from mimic_master.models.embeddings import DenseAndSparseEmbeddings, SparseVector


@pytest.fixture
def embedding_service():
    """Create a fresh embedding service instance for testing."""
    return EmbeddingService()


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
    """Test that sparse embeddings are generated in mock mode."""
    # Use mock directly to test sparse embeddings
    response = embedding_service._mock_embed(["fireball spell damage"])

    assert len(response.embeddings) == 1
    emb: DenseAndSparseEmbeddings = response.embeddings[0]

    assert len(emb.dense) > 0
    assert isinstance(emb.sparse, SparseVector)
    assert len(emb.sparse.indices) > 0
    assert len(emb.sparse.values) > 0


@pytest.mark.asyncio
async def test_embedding_consistency(embedding_service):
    """Test that same text produces same embedding in mock mode."""
    text = "D&D 5E rules"

    response1 = embedding_service._mock_embed([text])
    response2 = embedding_service._mock_embed([text])

    # In mock mode, same text should produce same embedding
    assert response1.embeddings[0].dense == response2.embeddings[0].dense
