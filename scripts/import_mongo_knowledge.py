"""Import processed exact-lookup knowledge documents into MongoDB."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne

from mimic_master.config import settings

DEFAULT_INPUT = Path("knowledge/mongo/entities.jsonl")
DEFAULT_COLLECTION = "knowledge_entities"


async def import_documents(
    collection: Any, docs: Iterable[dict[str, Any]], batch_size: int
) -> dict[str, int]:
    """Upsert documents into MongoDB by `_id`."""

    totals = {"matched": 0, "modified": 0, "upserted": 0}
    batch: list[UpdateOne] = []
    for doc in docs:
        batch.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
        if len(batch) >= batch_size:
            await _flush(collection, batch, totals)
            batch = []
    if batch:
        await _flush(collection, batch, totals)
    return totals


async def create_indexes(collection: Any) -> None:
    """Create indexes used by exact lookup and filtered retrieval."""

    await collection.create_index("aliases")
    await collection.create_index([("source_book", 1), ("record_type", 1)])
    await collection.create_index([("record_type", 1), ("title", 1)])
    await collection.create_index([("record_type", 1), ("eng_title", 1)])
    await collection.create_index(
        [("record_type", 1), ("numeric_filters.spell_level", 1)]
    )
    await collection.create_index([("record_type", 1), ("numeric_filters.cr", 1)])
    await collection.create_index("mechanic_tags")
    await collection.create_index("record_tags")


async def _flush(
    collection: Any, batch: list[UpdateOne], totals: dict[str, int]
) -> None:
    result = await collection.bulk_write(batch, ordered=False)
    totals["matched"] += result.matched_count
    totals["modified"] += result.modified_count
    totals["upserted"] += result.upserted_count


async def iter_jsonl(path: Path) -> AsyncIterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


async def import_file(
    *,
    input_path: Path,
    collection_name: str,
    batch_size: int,
    drop: bool,
) -> dict[str, int]:
    client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
        database = client[settings.mongodb_database]
        collection = database[collection_name]
        if drop:
            await collection.drop()
        await create_indexes(collection)
        docs = [doc async for doc in iter_jsonl(input_path)]
        totals = await import_documents(collection, docs, batch_size=batch_size)
        totals["total_documents"] = await collection.count_documents({})
        return totals
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--drop", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    totals = asyncio.run(
        import_file(
            input_path=args.input,
            collection_name=args.collection,
            batch_size=args.batch_size,
            drop=args.drop,
        )
    )
    print("Mongo knowledge import summary")
    for key, value in totals.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
