"""E2E smoke test for the Agent API surface."""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.helpers import require_run_e2e

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_agent_api_processes_request() -> None:
    """Prove the FastAPI Agent endpoint can process one request."""
    require_run_e2e()

    from mimic_master.api.app import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/agent/with-context",
            json={
                "query": "用一句话描述酒馆里玩家看到的场景。",
                "session_id": "e2e-agent-smoke",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["session_id"] == "e2e-agent-smoke"
    assert isinstance(payload["response"], str)
    assert payload["response"].strip()
    assert isinstance(payload["retrieved_context"], list)

    if os.getenv("REQUIRE_REAL_AGENT_LLM") == "1":
        assert "full implementation" not in payload["response"].lower()
