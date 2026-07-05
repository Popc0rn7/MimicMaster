"""Embed processed semantic chunks and upsert them to Pinecone."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.services.pinecone_service import get_pinecone_service

DEFAULT_INPUT = Path("knowledge/pinecone/chunks.jsonl")
DEFAULT_NAMESPACE = "rules_semantic"


def load_chunks(
    path: Path, limit: int | None = None, offset: int = 0
) -> list[dict[str, Any]]:
    """Load semantic chunks from JSONL."""

    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if line_number < offset:
                continue
            if not line.strip():
                continue
            chunks.append(json.loads(line))
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def normalize_proxy_environment(socksio_available: bool | None = None) -> list[str]:
    """Remove SOCKS all-proxy settings when httpx lacks SOCKS support."""

    if socksio_available is None:
        socksio_available = importlib.util.find_spec("socksio") is not None
    if socksio_available:
        return []

    removed = []
    for key in ("all_proxy", "ALL_PROXY"):
        value = os.environ.get(key, "")
        if value.lower().startswith(("socks4://", "socks5://")):
            os.environ.pop(key, None)
            removed.append(key)
    return removed


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build Pinecone-compatible flat metadata for a semantic chunk."""

    source_trace = chunk.get("source_trace", {})
    numeric_filters = chunk.get("numeric_filters", {})
    metadata: dict[str, Any] = {
        "chunk_id": chunk["_id"],
        "normalized_id": chunk.get("normalized_id", ""),
        "retrieval_mode": chunk.get("retrieval_mode", "pinecone_semantic"),
        "source_book": chunk.get("source_book", ""),
        "record_type": chunk.get("record_type", ""),
        "content_kind": chunk.get("content_kind", ""),
        "chunk_kind": chunk.get("chunk_kind", ""),
        "title": chunk.get("title", ""),
        "eng_title": chunk.get("eng_title") or "",
        "section_path": [str(item) for item in chunk.get("section_path", [])],
        "semantic_tags": [str(item) for item in chunk.get("semantic_tags", [])],
        "mechanic_tags": [str(item) for item in chunk.get("mechanic_tags", [])],
        "entity_refs": [str(item) for item in chunk.get("entity_refs", [])],
        "source_file": source_trace.get("source_file", ""),
        "source_key": source_trace.get("source_key", ""),
        "source_commit": source_trace.get("source_commit", ""),
        "chunk_index": int(chunk.get("chunk_index", 0)),
    }
    if chunk.get("page") is not None:
        metadata["page"] = int(chunk["page"])
    if "spell_level" in numeric_filters:
        metadata["spell_level"] = numeric_filters["spell_level"]
    if "cr" in numeric_filters:
        metadata["cr"] = str(numeric_filters["cr"])
    return metadata


async def index_chunks(
    chunks: list[dict[str, Any]],
    *,
    embedding_service: Any,
    pinecone_service: Any,
    namespace: str,
    batch_size: int,
    debug_single: bool = False,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> int:
    """Embed and upsert chunks in batches."""

    count = 0
    if debug_single:
        batch_size = 1
    batch_starts = range(0, len(chunks), batch_size)
    progress = tqdm(
        batch_starts,
        total=(len(chunks) + batch_size - 1) // batch_size,
        desc="Indexing Pinecone chunks",
        unit="batch",
    )
    for start in progress:
        batch = chunks[start : start + batch_size]
        ids = [chunk["_id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadata = [build_metadata(chunk) for chunk in batch]
        try:
            response = await retry_async(
                lambda: embedding_service.embed(texts),
                retries=retries,
                retry_delay=retry_delay,
                label="embedding",
                batch=batch,
                start=start,
            )
        except Exception as exc:
            raise RuntimeError(
                "Embedding failed for:\n" + describe_batch(batch, start=start)
            ) from exc
        embeddings = [
            embedding.model_dump() if hasattr(embedding, "model_dump") else embedding
            for embedding in response.embeddings
        ]
        try:
            await retry_async(
                lambda: pinecone_service.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    contents=texts,
                    metadata=metadata,
                    namespace=namespace,
                ),
                retries=retries,
                retry_delay=retry_delay,
                label="pinecone upsert",
                batch=batch,
                start=start,
            )
        except Exception as exc:
            raise RuntimeError(
                "Pinecone upsert failed for:\n" + describe_batch(batch, start=start)
            ) from exc
        count += len(batch)
        progress.set_postfix(indexed=count, chunks=len(chunks))
    return count


async def retry_async(
    operation: Callable[[], Awaitable[Any]],
    *,
    retries: int,
    retry_delay: float,
    label: str,
    batch: list[dict[str, Any]],
    start: int,
) -> Any:
    """Retry transient async operations with compact batch diagnostics."""

    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts:
                raise
            print(
                f"{label} attempt {attempt}/{attempts} failed: {exc}. "
                f"Retrying in {retry_delay:g}s.\n{describe_batch(batch, start=start)}"
            )
            await asyncio.sleep(retry_delay)
    raise RuntimeError(f"{label} retry loop exited unexpectedly")


def describe_batch(batch: list[dict[str, Any]], *, start: int) -> str:
    """Return compact diagnostics for a chunk batch."""

    lines = [f"batch_start={start} batch_size={len(batch)}"]
    for offset, chunk in enumerate(batch):
        text = chunk.get("text", "")
        lines.append(
            "  "
            f"offset={offset} "
            f"id={chunk.get('_id', '')} "
            f"source={chunk.get('source_book', '')} "
            f"record_type={chunk.get('record_type', '')} "
            f"title={chunk.get('title', '')} "
            f"chars={len(text)}"
        )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    removed_proxy_vars = normalize_proxy_environment()
    if removed_proxy_vars:
        print(
            "Removed SOCKS proxy env without socksio support: "
            + ", ".join(removed_proxy_vars)
        )
    chunks = load_chunks(args.input, limit=args.limit, offset=args.offset)
    if not chunks:
        print(f"No chunks loaded from {args.input}")
        return 0
    print(f"Loaded {len(chunks)} chunks from {args.input}")
    print(f"Offset: {args.offset}")
    print(f"Namespace: {args.namespace}")
    return await index_chunks(
        chunks,
        embedding_service=get_embedding_service(),
        pinecone_service=get_pinecone_service(),
        namespace=args.namespace,
        batch_size=args.batch_size,
        debug_single=args.debug_single,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--debug-single",
        action="store_true",
        help="Embed/upsert one chunk at a time and include chunk IDs in failures.",
    )
    return parser


def main() -> None:
    count = asyncio.run(run(build_parser().parse_args()))
    print(f"Indexed total: {count}")


if __name__ == "__main__":
    main()
