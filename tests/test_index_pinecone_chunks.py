"""Tests for indexing semantic Pinecone chunks."""

from pathlib import Path

import pytest

from scripts.index_pinecone_chunks import (
    build_metadata,
    describe_batch,
    index_chunks,
    load_chunks,
    normalize_proxy_environment,
)


def test_build_metadata_flattens_nested_chunk_fields() -> None:
    chunk = {
        "_id": "pinecone:phb:rule:cover:0",
        "source_book": "PHB",
        "record_type": "rule_section",
        "content_kind": "narrative",
        "chunk_kind": "rule_text",
        "title": "掩护",
        "eng_title": "Cover",
        "page": 196,
        "section_path": ["战斗", "掩护"],
        "semantic_tags": ["combat", "rule_text"],
        "mechanic_tags": ["attack"],
        "entity_refs": ["掩护", "Cover"],
        "numeric_filters": {"page": 196},
        "source_trace": {"source_file": "data/book/book-phb.json"},
        "normalized_id": "phb:rule:cover:0",
        "chunk_index": 0,
        "text": "掩护规则。",
    }

    metadata = build_metadata(chunk)

    assert metadata["source_book"] == "PHB"
    assert metadata["section_path"] == ["战斗", "掩护"]
    assert metadata["page"] == 196
    assert metadata["source_file"] == "data/book/book-phb.json"
    assert "numeric_filters" not in metadata
    assert "source_trace" not in metadata
    assert "text" not in metadata


def test_load_chunks_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"_id":"one","text":"一"}\n{"_id":"two","text":"二"}\n',
        encoding="utf-8",
    )

    chunks = load_chunks(path, limit=1)

    assert chunks == [{"_id": "one", "text": "一"}]


def test_load_chunks_respects_offset(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"_id":"one","text":"一"}\n{"_id":"two","text":"二"}\n',
        encoding="utf-8",
    )

    chunks = load_chunks(path, offset=1)

    assert chunks == [{"_id": "two", "text": "二"}]


def test_describe_batch_includes_ids_titles_and_lengths() -> None:
    batch = [
        {
            "_id": "one",
            "title": "简介",
            "record_type": "dm_section",
            "source_book": "DMG",
            "text": "abc",
        }
    ]

    description = describe_batch(batch, start=0)

    assert "batch_start=0" in description
    assert "id=one" in description
    assert "title=简介" in description
    assert "chars=3" in description


def test_normalize_proxy_environment_removes_socks_all_proxy_without_socksio(
    monkeypatch,
) -> None:
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:7897")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7897")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7897")

    removed = normalize_proxy_environment(socksio_available=False)

    assert removed == ["all_proxy", "ALL_PROXY"]
    assert "all_proxy" not in __import__("os").environ
    assert "ALL_PROXY" not in __import__("os").environ
    assert __import__("os").environ["https_proxy"] == "http://127.0.0.1:7897"


@pytest.mark.asyncio
async def test_index_chunks_embeds_and_upserts_batches() -> None:
    chunks = [
        {
            "_id": "one",
            "text": "一",
            "source_book": "PHB",
            "record_type": "rule_section",
        },
        {"_id": "two", "text": "二", "source_book": "DMG", "record_type": "dm_section"},
    ]
    embedding_service = FakeEmbeddingService()
    pinecone_service = FakePineconeService()

    count = await index_chunks(
        chunks,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        namespace="rules_semantic",
        batch_size=1,
    )

    assert count == 2
    assert embedding_service.calls == [["一"], ["二"]]
    assert [call["namespace"] for call in pinecone_service.calls] == [
        "rules_semantic",
        "rules_semantic",
    ]
    assert pinecone_service.calls[0]["ids"] == ["one"]


@pytest.mark.asyncio
async def test_index_chunks_retries_embedding_failure() -> None:
    chunks = [
        {
            "_id": "one",
            "text": "一",
            "source_book": "PHB",
            "record_type": "rule_section",
        }
    ]
    embedding_service = FailsOnceEmbeddingService()
    pinecone_service = FakePineconeService()

    count = await index_chunks(
        chunks,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        namespace="rules_semantic",
        batch_size=1,
        retries=1,
        retry_delay=0,
    )

    assert count == 1
    assert embedding_service.attempts == 2


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)

        class Embedding:
            def __init__(self) -> None:
                self.dense = [0.1, 0.2]
                self.sparse = {"indices": [], "values": []}

            def model_dump(self):
                return {"dense": self.dense, "sparse": self.sparse}

        class Response:
            embeddings = [Embedding() for _ in texts]

        return Response()


class FailsOnceEmbeddingService(FakeEmbeddingService):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def embed(self, texts):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("temporary provider failure")
        return await super().embed(texts)


class FakePineconeService:
    def __init__(self) -> None:
        self.calls = []

    async def upsert(self, *, ids, embeddings, contents, metadata, namespace):
        self.calls.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "contents": contents,
                "metadata": metadata,
                "namespace": namespace,
            }
        )
