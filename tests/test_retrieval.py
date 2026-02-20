"""Tests for retrieval and upsert functionality."""

import pytest

from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.models.embeddings import SparseVector


@pytest.mark.asyncio
async def test_retrieve_query():
    """Test basic retrieval query."""
    service = get_pinecone_service()
    embedding_service = get_embedding_service()

    # Generate query embedding
    embedding_response = await embedding_service.embed(["fireball spell"])
    query_embedding = embedding_response.embeddings[0].dense

    # Query (this will use mock/embedding service internally)
    try:
        results = await service.query(
            query_embedding=query_embedding,
            top_k=5,
            namespace="test",
        )
        # Results may be empty in mock mode, that's OK
        assert isinstance(results, list)
    except RuntimeError as e:
        # Expected if Pinecone not configured
        assert "not configured" in str(e)


@pytest.mark.asyncio
async def test_retrieve_with_sparse():
    """Test retrieval with sparse vector (hybrid search)."""
    service = get_pinecone_service()
    embedding_service = get_embedding_service()

    # Generate query embedding
    embedding_response = await embedding_service.embed(["opportunity attack"])
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

    try:
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
    embedding_service = get_embedding_service()

    # Generate embeddings
    embedding_response = await embedding_service.embed([
        "Longsword deals 1d8 slashing damage.",
        "Shortsword deals 1d6 piercing damage.",
    ])

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
