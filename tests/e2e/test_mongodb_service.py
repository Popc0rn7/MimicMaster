"""E2E smoke tests for the configured MongoDB knowledge store."""

from __future__ import annotations

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from scripts.import_mongo_knowledge import DEFAULT_COLLECTION
from tests.e2e.helpers import MONGO_ENTITIES_PATH, read_first_jsonl, require_run_e2e

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_mongodb_knowledge_collection_has_imported_entities() -> None:
    """Verify MongoDB is reachable and contains a known processed entity."""
    require_run_e2e()

    from mimic_master.config import settings

    expected = read_first_jsonl(MONGO_ENTITIES_PATH)
    client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
        collection = client[settings.mongodb_database][DEFAULT_COLLECTION]
        count = await collection.count_documents({})
        assert count > 0, f"MongoDB collection {DEFAULT_COLLECTION!r} is empty."

        found = await collection.find_one({"_id": expected["_id"]})
        assert found is not None, f"Expected Mongo entity {expected['_id']!r}."
        assert found["source_book"] in {"PHB", "DMG", "MM"}
        assert found["record_type"] == expected["record_type"]
        assert found["retrieval_mode"] == "mongo_exact"
    finally:
        client.close()
