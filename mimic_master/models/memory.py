"""Memory-related Pydantic models."""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    """A single dialogue turn in conversation history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SceneState(BaseModel):
    """Current scene state for game session."""

    location: str = Field(default="", description="Current location")
    time_of_day: str = Field(default="unknown", description="Time in-game")
    weather: str = Field(default="clear", description="Weather conditions")
    active_npcs: List[str] = Field(default_factory=list, description="Active NPC names")
    notes: str = Field(default="", description="GM notes about scene")


class CharacterState(BaseModel):
    """Player character state."""

    name: str = Field(..., description="Character name")
    hp: int = Field(..., description="Current hit points")
    max_hp: int = Field(..., description="Maximum hit points")
    level: int = Field(default=1, description="Character level")
    class_name: str = Field(default="", description="Character class")
    conditions: List[str] = Field(default_factory=list, description="Active conditions")
    spell_slots: Dict[int, int] = Field(default_factory=dict, description="Spell slots by level")


class GameState(BaseModel):
    """Overall game state maintained in working memory."""

    scene: SceneState = Field(default_factory=SceneState)
    players: Dict[str, CharacterState] = Field(default_factory=dict)
    session_notes: List[str] = Field(default_factory=list)


class Episode(BaseModel):
    """An episodic memory entry (session summary)."""

    id: str = Field(..., description="Unique episode ID")
    session_id: str = Field(..., description="Source session ID")
    summary: str = Field(..., description="Episode summary text")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    key_events: List[str] = Field(default_factory=list, description="Key events in episode")
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval")


class RetrievalResult(BaseModel):
    """Result from a retrieval operation."""

    id: str = Field(..., description="Document/episode ID")
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssembledContext(BaseModel):
    """Fully assembled context ready for LLM."""

    system_prompt: str = Field(..., description="System prompt with persona")
    state_context: str = Field(..., description="Current game state")
    dialogue_history: str = Field(..., description="Recent dialogue turns")
    knowledge_context: str = Field(default="", description="Retrieved rules/knowledge")
    episodic_context: str = Field(default="", description="Retrieved history/episodes")
    user_query: str = Field(..., description="Current user query")


class Intent(BaseModel):
    """Intent classification result."""

    type: str = Field(..., description="Intent type: QUERY_RULES, PROCEED_STORY, CHAT, COMBAT, etc.")
    confidence: float = Field(default=1.0, description="Confidence score")
