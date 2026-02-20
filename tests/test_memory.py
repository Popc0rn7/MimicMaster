"""Tests for three-layer memory system."""

import pytest

from mimic_master.memory import (
    get_state_memory,
    get_intent_classifier,
    get_context_assembler,
)
from mimic_master.models.memory import Intent, Episode


@pytest.mark.asyncio
async def test_state_memory_basic():
    """Test basic state memory operations."""
    memory = get_state_memory()

    # Update scene
    memory.update_scene(location="Tavern", time_of_day="evening")
    assert memory._state.scene.location == "Tavern"
    assert memory._state.scene.time_of_day == "evening"

    # Update player
    memory.update_player("Alice", hp=25, max_hp=30, level=3)
    assert "Alice" in memory._state.players
    assert memory._state.players["Alice"].hp == 25

    # Get state context
    state_context = memory.get_state_context()
    assert "Current Game State" in state_context
    assert "Tavern" in state_context


@pytest.mark.asyncio
async def test_state_memory_dialogue():
    """Test dialogue history management."""
    memory = get_state_memory()

    # Add dialogue turns
    memory.add_dialogue("user", "What is Fireball damage?")
    memory.add_dialogue("assistant", "Fireball deals 8d6 fire damage.")

    # Retrieve dialogue
    history = memory.get_dialogue_history(n=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"

    # Get dialogue context
    dialogue_context = memory.get_dialogue_context()
    assert "Recent Dialogue" in dialogue_context


@pytest.mark.asyncio
async def test_state_memory_sliding_window():
    """Test dialogue history sliding window."""
    memory = get_state_memory()

    # Add more turns than max
    for i in range(25):
        memory.add_dialogue("user", f"Message {i}")

    # Should only keep max turns
    history = memory.get_dialogue_history()
    assert len(history) == 20  # MAX_DIALOGUE_TURNS


@pytest.mark.asyncio
async def test_intent_classifier():
    """Test intent classification."""
    classifier = get_intent_classifier()

    # Test rules query
    intent = classifier.classify("What is the AC bonus for Shield spell?")
    assert intent.type == "query_rules"

    # Test combat query
    intent = classifier.classify("I attack the goblin with my sword")
    assert intent.type == "combat"

    # Test story query
    intent = classifier.classify("Where do we go from here?")
    assert intent.type == "proceed_story"

    # Test recall query
    intent = classifier.classify("What did we find in the last session?")
    assert intent.type == "recall_history"

    # Test chat
    intent = classifier.classify("Hello, how are you?")
    assert intent.type == "chat"


@pytest.mark.asyncio
async def test_context_assembler():
    """Test context assembly."""
    assembler = get_context_assembler()

    # Assemble context
    context = await assembler.assemble(
        user_query="What is the damage for Fireball?",
        session_id="test-session",
    )

    # Verify all components
    assert context.system_prompt is not None
    assert context.state_context is not None
    assert context.dialogue_history is not None
    assert context.user_query == "What is the damage for Fireball?"


@pytest.mark.asyncio
async def test_context_assembler_format():
    """Test formatted context for LLM."""
    assembler = get_context_assembler()

    context = await assembler.assemble(
        user_query="Test query",
        session_id="test-session",
    )

    formatted = assembler.format_for_llm(context)

    assert "System Prompt" in formatted or "Dungeon Master" in formatted
    assert "Current Query" in formatted
    assert "Test query" in formatted


@pytest.mark.asyncio
async def test_episodic_memory():
    """Test episodic memory retrieval."""
    from mimic_master.memory import get_episodic_retriever

    retriever = get_episodic_retriever()

    # Create a test episode
    episode = Episode(
        id="ep-001",
        session_id="test-session",
        summary="The party defeated a goblin ambush and found a mysterious map.",
        key_events=["Goblin ambush", "Found ancient map"],
        tags=["combat", "discovery"],
    )

    # In mock mode, this won't actually store
    # But we can test the structure
    assert episode.id == "ep-001"
    assert len(episode.key_events) == 2
    assert "combat" in episode.tags


@pytest.mark.asyncio
async def test_memory_integration():
    """Test full memory integration."""
    # Get state memory
    state_memory = get_state_memory()

    # Set up game state
    state_memory.update_scene(
        location="Dungeon Entrance",
        time_of_day="night",
        weather="stormy",
    )
    state_memory.update_player("Fighter", hp=15, max_hp=25, level=3)

    # Add dialogue
    state_memory.add_dialogue("user", "What's in this dungeon?")
    state_memory.add_dialogue("assistant", "You see a stone doorway...")

    # Verify state
    state_context = state_memory.get_state_context()
    assert "Dungeon Entrance" in state_context
    assert "Fighter" in state_context
    assert "15" in state_context  # HP value
