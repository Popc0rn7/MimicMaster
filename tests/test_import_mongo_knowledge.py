"""Tests for importing processed knowledge documents into MongoDB."""

import pytest

from scripts.import_mongo_knowledge import create_indexes, import_documents


class FakeCollection:
    def __init__(self) -> None:
        self.operations = []
        self.indexes = []

    async def bulk_write(self, operations, ordered):
        self.operations.extend(operations)
        self.ordered = ordered

        class Result:
            upserted_count = 1
            modified_count = 1
            matched_count = 1

        return Result()

    async def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))


@pytest.mark.asyncio
async def test_import_documents_upserts_by_id() -> None:
    collection = FakeCollection()
    docs = [
        {"_id": "phb:spell:fireball", "title": "火球术"},
        {"_id": "mm:monster:goblin", "title": "地精"},
    ]

    result = await import_documents(collection, docs, batch_size=1)

    assert result == {"matched": 2, "modified": 2, "upserted": 2}
    assert len(collection.operations) == 2
    first = collection.operations[0]
    assert first._filter == {"_id": "phb:spell:fireball"}
    assert first._doc["$set"]["title"] == "火球术"
    assert collection.ordered is False


@pytest.mark.asyncio
async def test_create_indexes_supports_exact_lookup_filters() -> None:
    collection = FakeCollection()

    await create_indexes(collection)

    index_keys = [keys for keys, _ in collection.indexes]
    assert "aliases" in index_keys
    assert [("source_book", 1), ("record_type", 1)] in index_keys
    assert [("record_type", 1), ("numeric_filters.spell_level", 1)] in index_keys
