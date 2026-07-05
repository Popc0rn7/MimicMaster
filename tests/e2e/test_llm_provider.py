"""E2E smoke test for an OpenAI-compatible LLM provider."""

from __future__ import annotations

import os

import pytest
from openai import OpenAI

from tests.e2e.helpers import require_run_e2e

pytestmark = pytest.mark.e2e


def test_openai_compatible_llm_provider_completes_chat() -> None:
    """Call a configured chat-completions provider once."""
    require_run_e2e()

    api_key = os.getenv("LLM_PROVIDER_API_KEY")
    base_url = os.getenv("LLM_PROVIDER_BASE_URL")
    model = os.getenv("LLM_PROVIDER_MODEL")
    if not (api_key and base_url and model):
        pytest.skip(
            "Set LLM_PROVIDER_API_KEY, LLM_PROVIDER_BASE_URL, and "
            "LLM_PROVIDER_MODEL to test the LLM provider."
        )

    from mimic_master.services.embedding_service import normalize_proxy_environment

    normalize_proxy_environment()
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly two words: service online",
            }
        ],
        max_tokens=16,
    )

    content = response.choices[0].message.content
    assert content is not None
    assert content.strip()
