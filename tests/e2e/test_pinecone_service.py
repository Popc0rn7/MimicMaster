"""E2E smoke tests for the configured Pinecone index."""

from __future__ import annotations

import pytest

from scripts.index_pinecone_chunks import DEFAULT_NAMESPACE
from tests.e2e.helpers import PINECONE_CHUNKS_PATH, read_first_jsonl, require_run_e2e

pytestmark = pytest.mark.e2e


def test_pinecone_rules_namespace_has_indexed_chunks() -> None:
    """Verify the current semantic namespace contains a known chunk ID."""
    require_run_e2e()

    from mimic_master.config import settings
    from mimic_master.services.pinecone_service import get_pinecone_service

    if not settings.is_pinecone_configured:
        pytest.skip("Pinecone not configured. Set PINECONE_API_KEY and PINECONE_INDEX.")

    chunk = read_first_jsonl(PINECONE_CHUNKS_PATH)
    index = get_pinecone_service().client.Index(settings.pinecone_index)

    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", {}) or {}
    namespace_stats = namespaces.get(DEFAULT_NAMESPACE)
    vector_count = getattr(namespace_stats, "vector_count", 0)
    assert vector_count > 0, f"Pinecone namespace {DEFAULT_NAMESPACE!r} is empty."

    fetched = index.fetch(ids=[chunk["_id"]], namespace=DEFAULT_NAMESPACE)
    vectors = getattr(fetched, "vectors", {}) or {}
    assert (
        chunk["_id"] in vectors
    ), f"Expected indexed chunk {chunk['_id']!r} in namespace {DEFAULT_NAMESPACE!r}."
