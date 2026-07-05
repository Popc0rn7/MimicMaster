"""Pytest configuration and fixtures."""

import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    if (
        os.getenv("RUN_REAL_EMBEDDING") == "1"
        or os.getenv("RUN_AGENT_E2E") == "1"
        or os.getenv("RUN_SERVICE_E2E") == "1"
    ):
        yield
        return

    # Clear and set test environment variables
    os.environ["PINECONE_INDEX"] = "test-index"
    os.environ["EMBEDDING_DIMENSION"] = "1024"
    os.environ["EMBEDDING_BACKEND"] = "local"
    os.environ["EMBEDDING_MODEL"] = "test-local-model"
    os.environ["LOCAL_API_KEY"] = "test-local-key"
    os.environ["LOCAL_BASE_URL"] = "http://localhost:9999/v1"
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
