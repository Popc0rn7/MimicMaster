"""Tests for reranker service."""

import pytest

from mimic_master.services.reranker_service import get_reranker_service
from mimic_master.models.reranker import RerankResponse


@pytest.mark.asyncio
async def test_reranker_basic():
    """Test basic reranking."""
    service = get_reranker_service()
    query = "What is the damage for Fireball?"
    documents = [
        "Fireball is a 3rd-level evocation spell that deals 8d6 fire damage.",
        "Magic Missile deals 3d4 force damage.",
        "Shield grants +5 AC to the caster.",
    ]

    response = await service.rerank(query, documents, top_n=2)

    assert len(response.results) == 2
    assert len(response.scores) == 2
    assert all(isinstance(score, float) for score in response.scores)
    assert all(isinstance(idx, int) for idx in response.results)
    # Scores should be in descending order
    assert response.scores[0] >= response.scores[1]


@pytest.mark.asyncio
async def test_reranker_all_results():
    """Test reranking without top_n (returns all results)."""
    service = get_reranker_service()
    query = "evocation spells"
    documents = [
        "Fireball is a 3rd-level evocation spell.",
        "Lightning Bolt is a 3rd-level evocation spell.",
        "Shield is an abjuration spell.",
    ]

    response = await service.rerank(query, documents)

    assert len(response.results) == 3
    assert len(response.scores) == 3
    assert len(response.results) == len(documents)


@pytest.mark.asyncio
async def test_reranker_empty_documents():
    """Test reranking with empty document list."""
    service = get_reranker_service()

    response = await service.rerank("test query", [], top_n=3)

    assert len(response.results) == 0
    assert len(response.scores) == 0


@pytest.mark.asyncio
async def test_reranker_top_n_larger_than_docs():
    """Test when top_n is larger than number of documents."""
    service = get_reranker_service()
    query = "test"
    documents = ["doc1", "doc2"]

    response = await service.rerank(query, documents, top_n=5)

    # Should only return at most 2 results
    assert len(response.results) == 2
    assert len(response.scores) == 2


@pytest.mark.asyncio
async def test_reranker_consistency():
    """Test that same inputs produce same outputs in mock mode."""
    service = get_reranker_service()
    query = "fireball damage"
    documents = ["Fireball deals 8d6 fire damage.", "Magic Missile deals 3d4 force damage."]

    response1 = await service.rerank(query, documents)
    response2 = await service.rerank(query, documents)

    # In mock mode, same inputs should produce same outputs
    assert response1.results == response2.results
    assert response1.scores == response2.scores
