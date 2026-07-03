"""Tests for retrieval and upsert functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.embedding_service import EmbeddingService


def fake_embed(texts):
    """Generate deterministic fake embeddings for retrieval tests."""
    return EmbeddingService()._mock_embed(texts)


@pytest.mark.asyncio
async def test_retrieve_query():
    """Test basic retrieval query."""
    service = get_pinecone_service()
    embedding_response = fake_embed(["fireball spell"])
    query_embedding = embedding_response.embeddings[0].dense

    # Query may fail if Pinecone is not configured; that is expected locally.
    try:
        results = await service.query(
            query_embedding=query_embedding,
            top_k=5,
            namespace="test",
        )
        assert isinstance(results, list)
    except RuntimeError as e:
        # Expected if Pinecone not configured
        assert "not configured" in str(e)


@pytest.mark.asyncio
async def test_retrieve_with_sparse():
    """Test retrieval with sparse vector (hybrid search)."""
    service = get_pinecone_service()
    embedding_response = fake_embed(["opportunity attack"])
    query_embedding = embedding_response.embeddings[0].dense
    sparse_vec = embedding_response.embeddings[0].sparse

    # Query with sparse vector
    try:
        results = await service.query(
            query_embedding=query_embedding,
            sparse_vector={"indices": sparse_vec.indices, "values": sparse_vec.values},
            top_k=5,
            namespace="test",
        )
        assert isinstance(results, list)
    except RuntimeError as e:
        # Expected if Pinecone not configured
        assert "not configured" in str(e)


@pytest.mark.asyncio
async def test_upsert_from_texts():
    """Test upserting documents from text."""
    service = get_pinecone_service()
    embedding_service = MagicMock()
    embedding_service.embed = AsyncMock(
        return_value=fake_embed(
            [
                "Fireball is a 3rd-level evocation spell that deals 8d6 fire damage.",
                "Magic Missile is a 1st-level evocation spell that deals 3d4 force damage.",
            ]
        )
    )

    try:
        with patch(
            "mimic_master.services.pinecone_service.get_embedding_service",
            return_value=embedding_service,
        ):
            await service.upsert_from_texts(
                ids=["test-001", "test-002"],
                texts=[
                    "Fireball is a 3rd-level evocation spell that deals 8d6 fire damage.",
                    "Magic Missile is a 1st-level evocation spell that deals 3d4 force damage.",
                ],
                namespace="test",
            )
        # If Pinecone is not configured, this will raise RuntimeError
    except RuntimeError as e:
        assert "not configured" in str(e)


@pytest.mark.asyncio
async def test_upsert_with_embeddings():
    """Test upserting documents with pre-computed embeddings."""
    service = get_pinecone_service()

    embedding_response = fake_embed(
        [
            "Longsword deals 1d8 slashing damage.",
            "Shortsword deals 1d6 piercing damage.",
        ]
    )

    try:
        await service.upsert(
            ids=["weapon-001", "weapon-002"],
            embeddings=embedding_response.embeddings,
            contents=[
                "Longsword deals 1d8 slashing damage.",
                "Shortsword deals 1d6 piercing damage.",
            ],
            metadata=[{"type": "weapon"}, {"type": "weapon"}],
            namespace="test",
        )
    except RuntimeError as e:
        assert "not configured" in str(e)
