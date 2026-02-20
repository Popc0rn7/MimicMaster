"""Tests for embedding service."""

import pytest

from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.models.embeddings import DenseAndSparseEmbeddings, SparseVector


@pytest.mark.asyncio
async def test_embedding_service():
    """Test basic embedding generation."""
    service = get_embedding_service()
    response = await service.embed(["Hello, world!"])

    assert len(response.embeddings) == 1
    assert len(response.embeddings[0].dense) > 0
    assert response.dense_dimension > 0


@pytest.mark.asyncio
async def test_embedding_multiple_texts():
    """Test embedding multiple texts."""
    service = get_embedding_service()
    response = await service.embed(["First text", "Second text"])

    assert len(response.embeddings) == 2
    assert len(response.embeddings[0].dense) > 0
    assert len(response.embeddings[1].dense) > 0


@pytest.mark.asyncio
async def test_sparse_embedding():
    """Test that sparse embeddings are generated."""
    service = get_embedding_service()
    response = await service.embed(["fireball spell damage"])

    assert len(response.embeddings) == 1
    emb: DenseAndSparseEmbeddings = response.embeddings[0]

    assert len(emb.dense) > 0
    assert isinstance(emb.sparse, SparseVector)
    assert len(emb.sparse.indices) > 0
    assert len(emb.sparse.values) > 0


@pytest.mark.asyncio
async def test_embedding_consistency():
    """Test that same text produces same embedding."""
    service = get_embedding_service()
    text = "D&D 5E rules"

    response1 = await service.embed([text])
    response2 = await service.embed([text])

    # In mock mode, same text should produce same embedding
    assert response1.embeddings[0].dense == response2.embeddings[0].dense
