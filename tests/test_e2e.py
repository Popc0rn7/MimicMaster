"""
End-to-End Tests for Real Services

This module tests the complete retrieval pipeline from ingest to retrieve
using real embedding and reranker services (not mocks).

Tests use the 'e2e-test' namespace in Pinecone and retain data after testing.
"""

import pytest

from mimic_master.config import settings
from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.services.reranker_service import get_reranker_service
from mimic_master.services.pinecone_service import get_pinecone_service


# Test data: D&D 5E spells for testing
SAMPLE_SPELLS = [
    {
        "id": "spell-001",
        "text": "Fireball is a 3rd-level evocation spell that creates a burst of flame. It deals 8d6 fire damage in a 20-foot-radius sphere, with an additional 1d6 for each slot level above 3rd. A Dexterity saving throw halves the damage.",
        "metadata": {"type": "spell", "school": "evocation", "level": 3}
    },
    {
        "id": "spell-002",
        "text": "Magic Missile is a 1st-level evocation spell that creates three glowing darts of magical force. Each dart deals 1d4+1 force damage. For each spell slot above 1st, you create an additional dart.",
        "metadata": {"type": "spell", "school": "evocation", "level": 1}
    },
    {
        "id": "spell-003",
        "text": "Shield is an abjuration spell that creates an invisible barrier of magical force. It grants you a +5 bonus to AC, including against the triggering attack, and blocks magic missiles. The spell lasts until the start of your next turn.",
        "metadata": {"type": "spell", "school": "abjuration", "level": 1}
    },
    {
        "id": "spell-004",
        "text": "Lightning Bolt is a 3rd-level evocation spell that creates a line of lightning. It deals 8d6 lightning damage in a 100-foot-long line, 5 feet wide. A Dexterity saving throw halves the damage.",
        "metadata": {"type": "spell", "school": "evocation", "level": 3}
    },
    {
        "id": "spell-005",
        "text": "Cure Wounds is a 1st-level evocation spell that heals a creature you touch. The spell restores 1d8 + your spellcasting ability modifier hit points. For each spell slot above 1st, the healing increases by 1d8.",
        "metadata": {"type": "spell", "school": "evocation", "level": 1}
    },
]

# Test data: D&D 5E combat rules
SAMPLE_COMBAT_RULES = [
    {
        "id": "rule-001",
        "text": "Opportunity Attack: You can make an opportunity attack when a hostile creature that you can see moves out of your reach. To do so, you use your reaction to make one melee attack against the provoking creature.",
        "metadata": {"type": "rule", "category": "combat"}
    },
    {
        "id": "rule-002",
        "text": "Cover: A target with half cover has a +2 bonus to AC and Dexterity saving throws. A target with three-quarters cover has a +5 bonus to AC and Dexterity saving throws. A target with total cover can't be targeted.",
        "metadata": {"type": "rule", "category": "combat"}
    },
    {
        "id": "rule-003",
        "text": "Initiative: At the start of combat, each combatant makes a Dexterity check to determine their order in the initiative. The DM ranks the combatants from highest to lowest. Ties can be broken by a Dexterity modifier or DM discretion.",
        "metadata": {"type": "rule", "category": "combat"}
    },
]

E2E_NAMESPACE = "e2e-test"


@pytest.mark.asyncio
async def test_service_connectivity():
    """Test 1: Verify all services are reachable and properly configured."""
    # Skip if mock mode is enabled
    if settings.use_mock_embedding:
        pytest.skip("Mock embedding mode enabled. Set PROVIDER_BASE_URL to use real service.")
    if settings.use_mock_reranker:
        pytest.skip("Mock reranker mode enabled. Set PROVIDER_BASE_URL to use real service.")
    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured. Set PINECONE_API_KEY and PINECONE_INDEX.")

    # Test embedding service
    embedding_service = get_embedding_service()
    embed_response = await embedding_service.embed(["test"])

    assert len(embed_response.embeddings) == 1, "Should return one embedding"
    assert len(embed_response.embeddings[0].dense) == settings.embedding_dimension, \
        f"Dense dimension should be {settings.embedding_dimension}"
    assert isinstance(embed_response.embeddings[0].sparse.indices, list), "Sparse indices should be a list"
    assert isinstance(embed_response.embeddings[0].sparse.values, list), "Sparse values should be a list"

    # Test reranker service
    reranker_service = get_reranker_service()
    rerank_response = await reranker_service.rerank(
        query="test",
        documents=["test document"],
        top_n=1
    )

    assert len(rerank_response.results) == 1, "Should return one result"
    assert len(rerank_response.scores) == 1, "Should return one score"


@pytest.mark.asyncio
async def test_embedding_generation():
    """Test 2: Verify embedding generation produces valid dense + sparse vectors."""
    if settings.use_mock_embedding:
        pytest.skip("Mock embedding mode enabled.")

    embedding_service = get_embedding_service()

    test_texts = [
        "Fireball is a powerful spell",
        "Magic Missile deals force damage",
        "Shield grants defensive bonuses",
    ]

    response = await embedding_service.embed(test_texts)

    # Validate response structure
    assert len(response.embeddings) == len(test_texts), "Embedding count should match input"

    for i, emb in enumerate(response.embeddings):
        # Validate dense vector
        assert len(emb.dense) == settings.embedding_dimension, \
            f"Dense dimension mismatch for text {i+1}"
        assert all(isinstance(v, float) for v in emb.dense), \
            f"Dense vector should contain floats for text {i+1}"

        # Validate sparse vector
        assert len(emb.sparse.indices) == len(emb.sparse.values), \
            f"Sparse indices and values count mismatch for text {i+1}"
        assert all(isinstance(i, int) for i in emb.sparse.indices), \
            f"Sparse indices should be ints for text {i+1}"
        assert all(isinstance(v, float) for v in emb.sparse.values), \
            f"Sparse values should be floats for text {i+1}"


@pytest.mark.asyncio
async def test_reranking():
    """Test 3: Verify reranking produces relevant, sorted results."""
    if settings.use_mock_reranker:
        pytest.skip("Mock reranker mode enabled.")

    reranker_service = get_reranker_service()

    query = "What spells deal fire damage?"
    documents = [
        "Fireball is a 3rd-level evocation spell that deals 8d6 fire damage.",
        "Magic Missile deals 3d4 force damage.",
        "Shield grants +5 AC to the caster.",
        "Lightning Bolt is a 3rd-level evocation spell that deals 8d6 lightning damage.",
        "Cure Wounds heals a creature you touch.",
    ]

    # Test without top_n (real reranker may filter low-relevance results)
    response_all = await reranker_service.rerank(query, documents)
    # Real reranker may not return all documents (filters by relevance threshold)
    assert len(response_all.results) > 0, "Should return at least one result"
    assert len(response_all.results) <= len(documents), "Should not return more than input documents"

    # Verify scores are in descending order
    for i in range(len(response_all.scores) - 1):
        assert response_all.scores[i] >= response_all.scores[i+1], \
            "Scores should be in descending order"

    # Test with top_n
    response_top2 = await reranker_service.rerank(query, documents, top_n=2)
    assert len(response_top2.results) == 2, "Should return top 2 results"


@pytest.mark.asyncio
async def test_ingest_pipeline():
    """Test 4: Verify ingest pipeline (embedding + upsert to Pinecone)."""
    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured.")

    pinecone_service = get_pinecone_service()
    embedding_service = get_embedding_service()

    # Ingest spells
    ids = [spell["id"] for spell in SAMPLE_SPELLS]
    texts = [spell["text"] for spell in SAMPLE_SPELLS]
    metadata = [spell["metadata"] for spell in SAMPLE_SPELLS]

    # Generate embeddings
    embed_response = await embedding_service.embed(texts)
    assert len(embed_response.embeddings) == len(ids), "Embedding count should match input"

    # Upsert to Pinecone
    await pinecone_service.upsert(
        ids=ids,
        embeddings=embed_response.embeddings,
        contents=texts,
        metadata=metadata,
        namespace=E2E_NAMESPACE,
    )

    # Ingest rules using upsert_from_texts
    rule_ids = [rule["id"] for rule in SAMPLE_COMBAT_RULES]
    rule_texts = [rule["text"] for rule in SAMPLE_COMBAT_RULES]
    rule_metadata = [rule["metadata"] for rule in SAMPLE_COMBAT_RULES]

    await pinecone_service.upsert_from_texts(
        ids=rule_ids,
        texts=rule_texts,
        metadata=rule_metadata,
        namespace=E2E_NAMESPACE,
    )


@pytest.mark.asyncio
async def test_retrieve_pipeline_dense():
    """Test 5: Verify retrieve pipeline using dense vector only."""
    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured.")

    pinecone_service = get_pinecone_service()
    embedding_service = get_embedding_service()

    query = "What spells deal fire damage?"
    top_k = 3

    # Generate query embedding
    embed_response = await embedding_service.embed([query])
    query_embedding = embed_response.embeddings[0].dense

    # Query Pinecone
    results = await pinecone_service.query(
        query_embedding=query_embedding,
        top_k=top_k,
        sparse_vector=None,
        namespace=E2E_NAMESPACE,
    )

    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= top_k, f"Should return at most {top_k} results"

    # Validate result structure
    for doc in results:
        assert isinstance(doc.id, str), "Document ID should be a string"
        assert isinstance(doc.content, str), "Document content should be a string"
        assert isinstance(doc.score, float), "Document score should be a float"
        assert isinstance(doc.metadata, dict), "Document metadata should be a dict"


@pytest.mark.asyncio
async def test_retrieve_pipeline_hybrid():
    """Test 6: Verify retrieve pipeline using hybrid search (dense + sparse)."""
    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured.")

    pinecone_service = get_pinecone_service()
    embedding_service = get_embedding_service()

    query = "How does cover work in combat?"
    top_k = 3

    # Generate query embedding (dense + sparse)
    embed_response = await embedding_service.embed([query])
    query_embedding = embed_response.embeddings[0].dense
    sparse_vec = embed_response.embeddings[0].sparse

    # Query Pinecone with hybrid search
    results = await pinecone_service.query(
        query_embedding=query_embedding,
        top_k=top_k,
        sparse_vector={
            "indices": sparse_vec.indices,
            "values": sparse_vec.values,
        },
        namespace=E2E_NAMESPACE,
    )

    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= top_k, f"Should return at most {top_k} results"


@pytest.mark.asyncio
async def test_complete_pipeline():
    """Test 7: Verify complete pipeline with retrieval + reranking."""
    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured.")

    pinecone_service = get_pinecone_service()
    embedding_service = get_embedding_service()
    reranker_service = get_reranker_service()

    query = "What evocation spells are available at level 3?"
    top_k = 5

    # Step 1: Generate query embedding
    embed_response = await embedding_service.embed([query])
    query_embedding = embed_response.embeddings[0].dense

    # Step 2: Query Pinecone
    initial_results = await pinecone_service.query(
        query_embedding=query_embedding,
        top_k=top_k,
        sparse_vector=None,
        namespace=E2E_NAMESPACE,
    )

    # Skip reranking if no results
    if len(initial_results) == 0:
        pytest.skip("No initial results to rerank.")

    # Step 3: Rerank results
    doc_texts = [doc.content for doc in initial_results]
    rerank_response = await reranker_service.rerank(
        query=query,
        documents=doc_texts,
        top_n=len(initial_results),
    )

    assert len(rerank_response.results) == len(initial_results), \
        "Reranked count should match initial results"

    # Verify reranked results are valid indices
    for idx in rerank_response.results:
        assert 0 <= idx < len(initial_results), f"Invalid index: {idx}"

    # Verify scores are in descending order
    for i in range(len(rerank_response.scores) - 1):
        assert rerank_response.scores[i] >= rerank_response.scores[i+1], \
            "Rerank scores should be in descending order"


@pytest.mark.asyncio
async def test_retrieve_with_metadata_filter():
    """Test 8: Verify retrieval with metadata filtering."""
    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured.")

    pinecone_service = get_pinecone_service()
    embedding_service = get_embedding_service()

    query = "spells"
    top_k = 10

    # Generate query embedding
    embed_response = await embedding_service.embed([query])
    query_embedding = embed_response.embeddings[0].dense

    # Query with metadata filter for spells
    results = await pinecone_service.query(
        query_embedding=query_embedding,
        top_k=top_k,
        sparse_vector=None,
        filter_dict={"type": "spell"},
        namespace=E2E_NAMESPACE,
    )

    assert isinstance(results, list), "Results should be a list"

    # Verify all results have type="spell" in metadata
    for doc in results:
        if doc.metadata:
            assert doc.metadata.get("type") == "spell", \
                f"Document {doc.id} should have type='spell' in metadata"
