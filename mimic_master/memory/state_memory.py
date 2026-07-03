"""State & Working Memory Module.

Manages the current game state and recent dialogue history.
This information is always included in the LLM context without retrieval.
"""

from collections import deque
from typing import Optional, List

from mimic_master.models.memory import (
    DialogueTurn,
    SceneState,
    CharacterState,
    GameState,
)

# Constants
MAX_DIALOGUE_TURNS = 20  # Maximum dialogue turns to keep in history
MAX_SESSION_NOTES = 50  # Maximum session notes to keep


class StateMemory:
    """
    State and Working Memory for the DM Agent.

    Maintains:
    1. Current game state (scene, NPCs, player characters)
    2. Sliding window of recent dialogue history
    3. Session notes for tracking important events
    """

    def __init__(
        self,
        max_dialogue_turns: int = MAX_DIALOGUE_TURNS,
        max_session_notes: int = MAX_SESSION_NOTES,
    ) -> None:
        """
        Initialize the state memory.

        Args:
            max_dialogue_turns: Maximum number of dialogue turns to keep
            max_session_notes: Maximum number of session notes to keep
        """
        self._max_dialogue_turns = max_dialogue_turns
        self._max_session_notes = max_session_notes

        # Game state
        self._state: GameState = GameState()

        # Dialogue history (sliding window)
        self._dialogue_history: deque[DialogueTurn] = deque(maxlen=max_dialogue_turns)

    # State management

    @property
    def state(self) -> GameState:
        """Get the current game state."""
        return self._state

    def update_scene(self, **kwargs) -> None:
        """
        Update the scene state with the given fields.

        Args:
            **kwargs: Fields to update in SceneState
        """
        current_data = self._state.scene.model_dump()
        current_data.update(kwargs)
        self._state.scene = SceneState(**current_data)

    def update_player(self, player_name: str, **kwargs) -> None:
        """
        Update a player's character state.

        Args:
            player_name: Name of the player/character
            **kwargs: Fields to update in CharacterState
        """
        if player_name not in self._state.players:
            self._state.players[player_name] = CharacterState(
                name=player_name,
                hp=kwargs.get("hp", 0),
                max_hp=kwargs.get("max_hp", 0),
            )

        current_data = self._state.players[player_name].model_dump()
        current_data.update(kwargs)
        self._state.players[player_name] = CharacterState(**current_data)

    def add_session_note(self, note: str) -> None:
        """
        Add a note to the session notes.

        Args:
            note: The note to add
        """
        self._state.session_notes.append(note)
        if len(self._state.session_notes) > self._max_session_notes:
            self._state.session_notes.pop(0)

    def reset_state(self) -> None:
        """Reset the game state to defaults."""
        self._state = GameState()
        self._dialogue_history.clear()

    # Dialogue history management

    def add_dialogue(self, role: str, content: str) -> None:
        """
        Add a dialogue turn to the history.

        Args:
            role: 'user' or 'assistant'
            content: The message content
        """
        turn = DialogueTurn(role=role, content=content)
        self._dialogue_history.append(turn)

    def get_dialogue_history(self, n: Optional[int] = None) -> List[DialogueTurn]:
        """
        Get the most recent dialogue turns.

        Args:
            n: Number of turns to return (None for all)

        Returns:
            List of dialogue turns
        """
        history = list(self._dialogue_history)
        if n is not None:
            return history[-n:]
        return history

    def clear_dialogue_history(self) -> None:
        """Clear the dialogue history."""
        self._dialogue_history.clear()

    # Context serialization

    def get_state_context(self) -> str:
        """
        Serialize the current state to a markdown string.

        Returns:
            Markdown-formatted state context
        """
        lines = ["## Current Game State\n"]

        # Scene information
        scene = self._state.scene
        lines.append("### Scene")
        if scene.location:
            lines.append(f"- Location: {scene.location}")
        if scene.time_of_day != "unknown":
            lines.append(f"- Time: {scene.time_of_day}")
        if scene.weather != "clear":
            lines.append(f"- Weather: {scene.weather}")
        if scene.active_npcs:
            lines.append(f"- Active NPCs: {', '.join(scene.active_npcs)}")
        if scene.notes:
            lines.append(f"- Notes: {scene.notes}")
        lines.append("")

        # Player information
        if self._state.players:
            lines.append("### Characters")
            for player_name, char_state in self._state.players.items():
                lines.append(f"#### {player_name}")
                lines.append(f"- HP: {char_state.hp}/{char_state.max_hp}")
                lines.append(f"- Level {char_state.level} {char_state.class_name}")
                if char_state.conditions:
                    lines.append(f"- Conditions: {', '.join(char_state.conditions)}")
                if char_state.spell_slots:
                    slots_str = ", ".join(
                        f"Level {lvl}: {slots}"
                        for lvl, slots in sorted(char_state.spell_slots.items())
                    )
                    lines.append(f"- Spell Slots: {slots_str}")
            lines.append("")

        # Recent session notes
        if self._state.session_notes:
            lines.append("### Recent Notes")
            for note in self._state.session_notes[-5:]:  # Last 5 notes
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines)

    def get_dialogue_context(self, n: Optional[int] = None) -> str:
        """
        Serialize recent dialogue history to a markdown string.

        Args:
            n: Number of turns to include (None for all)

        Returns:
            Markdown-formatted dialogue history
        """
        history = self.get_dialogue_history(n)
        if not history:
            return ""

        lines = ["## Recent Dialogue\n"]

        for turn in history:
            role_label = "Player" if turn.role == "user" else "Dungeon Master"
            timestamp = turn.timestamp.strftime("%H:%M")
            lines.append(f"[{timestamp}] **{role_label}**: {turn.content}")

        return "\n".join(lines)


# Singleton instance
_state_memory: Optional[StateMemory] = None


def get_state_memory() -> StateMemory:
    """Get the singleton state memory instance."""
    global _state_memory
    if _state_memory is None:
        _state_memory = StateMemory()
    return _state_memory


def reset_state_memory() -> None:
    """Reset the singleton state memory instance."""
    global _state_memory
    _state_memory = None
