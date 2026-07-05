"""Shared helpers for real-service E2E smoke tests."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINECONE_CHUNKS_PATH = PROJECT_ROOT / "knowledge" / "pinecone" / "chunks.jsonl"
MONGO_ENTITIES_PATH = PROJECT_ROOT / "knowledge" / "mongo" / "entities.jsonl"


def require_run_e2e() -> None:
    """Skip unless the caller explicitly enabled real-service tests."""
    if os.getenv("RUN_E2E") != "1":
        pytest.skip("Set RUN_E2E=1 to call configured real services.")


def reload_config() -> Any:
    """Reload app config after tests opt into the real environment."""
    from mimic_master import config

    importlib.reload(config)
    return config


def read_first_jsonl(path: Path) -> dict[str, Any]:
    """Return the first JSON object from a JSONL fixture file."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)
    raise AssertionError(f"No JSONL records found in {path}")
