"""Pytest configuration and fixtures."""

import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Clear and set test environment variables
    os.environ["PINECONE_INDEX"] = "test-index"
    os.environ["EMBEDDING_DIMENSION"] = "1024"
    os.environ["EMBEDDING_PROVIDER_TYPE"] = "http"
    os.environ["EMBEDDING_PROVIDER_URL"] = "http://localhost:9999/embeddings"
    os.environ["RERANKER_PROVIDER_URL"] = "mock"

    # Clear any existing API keys
    os.environ.pop("PINECONE_API_KEY", None)

    # Force settings reimport
    from mimic_master import config
    import importlib

    importlib.reload(config)

    # Yield control to tests
    yield

    # Cleanup after all tests
    os.environ.pop("PINECONE_API_KEY", None)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Fireball is a 3rd-level evocation spell.",
        "Magic Missile deals 3d4 force damage.",
        "Shield grants +5 AC to the caster.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is the damage for Fireball?"
