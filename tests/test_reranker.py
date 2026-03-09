"""Tests for reranker service."""

import pytest

from mimic_master.config import settings
from mimic_master.services.reranker_service import RerankerService, get_reranker_service
from mimic_master.models.reranker import RerankResponse


@pytest.fixture
def reranker_service():
    """Create a fresh reranker service instance for testing."""
    return RerankerService()


@pytest.mark.asyncio
async def test_reranker_basic(reranker_service):
    """Test basic reranking using mock."""
    query = "What is the damage for Fireball?"
    documents = [
        "Fireball is a 3rd-level evocation spell that deals 8d6 fire damage.",
        "Magic Missile deals 3d4 force damage.",
        "Shield grants +5 AC to the caster.",
    ]

    # Use mock directly to avoid external API calls
    response = await reranker_service.rerank(query, documents, top_n=2)

    assert len(response.results) == 2
    assert len(response.scores) == 2
    assert all(isinstance(score, float) for score in response.scores)
    assert all(isinstance(idx, int) for idx in response.results)
    # Scores should be in descending order
    assert response.scores[0] >= response.scores[1]


@pytest.mark.asyncio
async def test_reranker_mock(reranker_service):
    """Test reranking without top_n (returns all results) using mock."""
    query = "evocation spells"
    documents = [
        "Fireball is a 3rd-level evocation spell.",
        "Lightning Bolt is a 3rd-level evocation spell.",
        "Shield is an abjuration spell.",
    ]

    response = await reranker_service.rerank(query, documents)

    assert len(response.results) == 3
    assert len(response.scores) == 3
    assert len(response.results) == len(documents)


@pytest.mark.asyncio
async def test_reranker_empty(reranker_service):
    """Test reranking with empty document list using mock."""

    response = await reranker_service.rerank("test query", [], top_n=3)

    assert len(response.results) == 0
    assert len(response.scores) == 0


@pytest.mark.asyncio
async def test_reranker_top_n_larger_than_docs(reranker_service):
    """Test when top_n is larger than number of documents using mock."""
    query = "test"
    documents = ["doc1", "doc2"]

    response = await reranker_service.rerank(query, documents, top_n=5)

    # Should only return at most 2 results
    assert len(response.results) == 2
    assert len(response.scores) == 2


@pytest.mark.asyncio
async def test_reranker_consistency(reranker_service):
    """Test that same inputs produce same outputs in mock mode."""
    query = "fireball damage"
    documents = [
        "Fireball deals 8d6 fire damage.",
        "Magic Missile deals 3d4 force damage.",
    ]

    # Use mock directly for consistency test
    response1 = await reranker_service.rerank(query, documents)
    response2 = await reranker_service.rerank(query, documents)

    # In mock mode, same inputs should produce same outputs
    assert response1.results == response2.results
    assert response1.scores == response2.scores
