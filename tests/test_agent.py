"""Tests for DM Agent functionality."""

import pytest

from mimic_master.core import DMAgent, DMAgentBuilder


@pytest.mark.asyncio
async def test_agent_process_query():
    """Test basic query processing."""
    agent = DMAgentBuilder().build()

    response = await agent.process_query(
        query="What is the damage for Fireball spell?",
        session_id="test-session",
    )

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_agent_process_with_context():
    """Test query processing with context."""
    agent = DMAgentBuilder().build()

    response, context = await agent.process_with_context(
        query="What is the damage for Fireball spell?",
        session_id="test-session",
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert context is not None
    assert context.system_prompt is not None
    assert context.state_context is not None
    assert context.dialogue_history is not None
    assert context.user_query == "What is the damage for Fireball spell?"


@pytest.mark.asyncio
async def test_agent_builder():
    """Test agent builder pattern."""
    agent = (DMAgentBuilder()
              .with_model("claude-3-opus")
              .build())

    assert agent.model_name == "claude-3-opus"


@pytest.mark.asyncio
async def test_agent_multiple_queries():
    """Test multiple sequential queries."""
    agent = DMAgentBuilder().build()

    queries = [
        "What is Fireball?",
        "How does opportunity attack work?",
        "What is Armor Class?",
    ]

    for query in queries:
        response = await agent.process_query(query)
        assert isinstance(response, str)
        assert len(response) > 0
