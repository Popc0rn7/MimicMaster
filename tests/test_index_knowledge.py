"""Tests for knowledge indexing script."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import json
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.index_knowledge import KnowledgeIndexer
from mimic_master.models.embeddings import (
    EmbeddingResponse,
    DenseAndSparseEmbeddings,
    SparseVector,
)
from mimic_master.models.memory import KnowledgeCategory, SourceBook


@pytest.fixture
def temp_knowledge_dir():
    """Create temporary knowledge directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        use_dir = base / "knowledge" / "use"
        use_dir.mkdir(parents=True)
        yield use_dir


@pytest.fixture
def sample_monster_data():
    """Sample monster JSONL data."""
    return [
        {
            "text": "Goblin\nSmall humanoid (goblinoid), Neutral Evil\nAC 15, HP 7, Speed 30 ft.\nCR 1/4 (50 XP)",
            "chapter": "Goblinoids",
        },
        {
            "text": "Dragon, Red\nHuge dragon, Chaotic Evil\nAC 19, HP 256, Speed 40 ft., Fly 80 ft.\nCR 17 (18,000 XP)",
            "chapter": "Dragons",
        },
    ]


@pytest.fixture
def sample_rule_data():
    """Sample rule JSONL data."""
    return [
        {
            "text": "Fireball\n3rd-level evocation\nCasting Time: 1 action\nRange: 150 feet\nComponents: V, S, M (a tiny ball of bat guano and sulfur)",
            "chapter": "Spells",
        },
    ]


@pytest.fixture
def embedding_service_mock():
    """Mock embedding service."""
    mock = MagicMock()
    mock.embed = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[
                DenseAndSparseEmbeddings(
                    dense=[0.1] * 1024,
                    sparse=SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2]),
                )
                for _ in range(2)
            ],
            dense_dimension=1024,
        )
    )
    return mock


@pytest.fixture
def pinecone_service_mock():
    """Mock pinecone service."""
    mock = MagicMock()
    mock.upsert = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_missing_processed_jsonl_points_to_external_placeholder(
    temp_knowledge_dir, tmp_path
):
    """Missing processed data should fail with a clear data-management hint."""
    with (
        patch("scripts.index_knowledge.USE_DIR", temp_knowledge_dir),
        patch("scripts.index_knowledge.BASE_DIR", tmp_path),
    ):
        indexer = KnowledgeIndexer()

        with pytest.raises(FileNotFoundError, match="knowledge/external"):
            await indexer.ensure_processed_jsonl("phb")


@pytest.mark.asyncio
async def test_empty_processed_jsonl_is_rejected(temp_knowledge_dir):
    """Empty processed data should not be silently indexed."""
    (temp_knowledge_dir / "rag_phb.jsonl").write_text("", encoding="utf-8")

    with patch("scripts.index_knowledge.USE_DIR", temp_knowledge_dir):
        indexer = KnowledgeIndexer()

        with pytest.raises(ValueError, match="empty"):
            await indexer.ensure_processed_jsonl("phb")


@pytest.mark.asyncio
async def test_index_monsters(
    temp_knowledge_dir,
    sample_monster_data,
    embedding_service_mock,
    pinecone_service_mock,
):
    """Test indexing monsters."""
    # Create test JSONL file
    jsonl_path = temp_knowledge_dir / "rag_mm.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sample_monster_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (
        patch("scripts.index_knowledge.USE_DIR", temp_knowledge_dir),
        patch("scripts.index_knowledge.get_embedding_service") as mock_get_embedding,
        patch("scripts.index_knowledge.get_pinecone_service") as mock_get_pinecone,
    ):
        mock_get_embedding.return_value = embedding_service_mock
        mock_get_pinecone.return_value = pinecone_service_mock

        indexer = KnowledgeIndexer()
        count = await indexer.index_monsters(limit=10)

        assert count == 2
        assert embedding_service_mock.embed.call_count == 1
        assert pinecone_service_mock.upsert.call_count == 1

        # Verify upsert was called with correct parameters
        call_args = pinecone_service_mock.upsert.call_args
        assert len(call_args.kwargs["ids"]) == 2
        assert call_args.kwargs["namespace"] == "monsters"


@pytest.mark.asyncio
async def test_index_rules_phb(
    temp_knowledge_dir, sample_rule_data, embedding_service_mock, pinecone_service_mock
):
    """Test indexing PHB rules."""
    jsonl_path = temp_knowledge_dir / "rag_phb.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sample_rule_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (
        patch("scripts.index_knowledge.USE_DIR", temp_knowledge_dir),
        patch("scripts.index_knowledge.get_embedding_service") as mock_get_embedding,
        patch("scripts.index_knowledge.get_pinecone_service") as mock_get_pinecone,
    ):
        mock_get_embedding.return_value = embedding_service_mock
        mock_get_pinecone.return_value = pinecone_service_mock

        indexer = KnowledgeIndexer()
        count = await indexer.index_rules("phb", limit=10)

        assert count == 1
        pinecone_service_mock.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_detect_image_in_text():
    """Test image detection in text."""
    from scripts.index_knowledge import KnowledgeIndexer

    indexer = KnowledgeIndexer()

    # Test with image tag
    text_with_img = "[IMG:images/goblin.jpg] Goblin Small humanoid"
    processed, image_path = indexer.detect_image_in_text(text_with_img)

    assert image_path == "images/goblin.jpg"
    assert "[图片占位符]" in processed

    # Test without image tag
    text_no_img = "Goblin Small humanoid"
    processed, image_path = indexer.detect_image_in_text(text_no_img)

    assert image_path is None
    assert processed == text_no_img


@pytest.mark.asyncio
async def test_parse_monster_data():
    """Test parsing monster metadata."""
    from scripts.index_knowledge import KnowledgeIndexer

    indexer = KnowledgeIndexer()

    text = "Dragon, Red Huge dragon, Chaotic Evil AC 19, HP 256"
    metadata = indexer.parse_monster_data(text, "Dragons")

    assert metadata.category == KnowledgeCategory.MONSTER
    assert metadata.source_book == SourceBook.MM
    assert metadata.chapter == "Dragons"
    assert metadata.name == "Dragon,"
    assert metadata.monster_type == "dragon"


@pytest.mark.asyncio
async def test_parse_rule_data():
    """Test parsing rule metadata."""
    from scripts.index_knowledge import KnowledgeIndexer

    indexer = KnowledgeIndexer()

    # Test spell detection
    text = "Fireball 3rd-level spell evocation"
    metadata = indexer.parse_rule_data(text, "Spells", SourceBook.PHB)

    assert metadata.category == KnowledgeCategory.RULE
    assert metadata.source_book == SourceBook.PHB
    assert metadata.rule_type == "spell"
