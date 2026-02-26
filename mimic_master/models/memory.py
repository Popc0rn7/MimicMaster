"""Memory-related Pydantic models."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    """A single dialogue turn in conversation history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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


class RetrievalConfig(BaseModel):
    """Configuration for retrieval operations."""

    top_k_rules: int = Field(default=5, description="Number of rules to retrieve")
    top_k_episodes: int = Field(default=3, description="Number of episodes to retrieve")
    rules_namespace: str = Field(default="rules", description="Namespace for rules")
    episodes_namespace: str = Field(default="episodes", description="Namespace for episodes")
    monsters_namespace: str = Field(default="monsters", description="Namespace for monsters")
    spells_namespace: str = Field(default="spells", description="Namespace for spells")
    use_rerank: bool = Field(default=True, description="Whether to use reranking")


class KnowledgeMetadata(BaseModel):
    """Metadata for knowledge base entries."""

    category: str = Field(..., description="Category: monster, rule, dm_guide, spell")
    source_book: str = Field(..., description="Source book: PHB, MM, DMG")
    chapter: str = Field(default="", description="Chapter in source book")
    # For monsters
    name: str = Field(default="", description="Monster/rule name")
    monster_type: str = Field(default="", description="Monster type (aberration, dragon, etc.)")
    size: str = Field(default="", description="Size (tiny, small, medium, large, huge, gargantuan)")
    alignment: str = Field(default="", description="Alignment")
    cr: str = Field(default="", description="Challenge rating")
    # For rules
    rule_type: str = Field(default="", description="Rule type: class, spell, race, background, feat, skill, equipment")
    section: str = Field(default="", description="Section name")
    # For DM guides
    dm_type: str = Field(default="", description="DM type: treasure, trap, encounter, variant")
    # Image
    has_image: bool = Field(default=False, description="Whether entry has associated image")
    image_path: str = Field(default="", description="Path to image file")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Pinecone metadata."""
        return {k: v for k, v in self.model_dump().items() if v}


# Knowledge category constants
class KnowledgeCategory:
    """Knowledge category constants."""

    MONSTER = "monster"
    RULE = "rule"
    DM_GUIDE = "dm_guide"
    SPELL = "spell"


# Source book constants
class SourceBook:
    """D&D source book constants."""

    PHB = "PHB"
    MM = "MM"
    DMG = "DMG"


# Monster type constants (partial list)
MONSTER_TYPES = [
    "aberration", "beast", "celestial", "construct", "dragon",
    "elemental", "fey", "fiend", "giant", "humanoid", "monstrosity",
    "ooze", "plant", "undead"
]


# Rule type constants
RULE_TYPES = [
    "class", "spell", "race", "background", "feat",
    "skill", "equipment", "rule"
]
