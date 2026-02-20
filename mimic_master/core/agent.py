"""Core DM Agent implementation.

Integrates with the three-layer memory system:
- State & Working Memory (current state + dialogue history)
- Static Knowledge Retrieval (rules via Hybrid Search + Rerank)
- Episodic Memory (past session summaries)
"""

from typing import Optional

from mimic_master.memory import (
    ContextAssembler,
    get_context_assembler,
    AssembledContext,
)


class DMAgent:
    """
    Dungeon Master Agent for D&D 5E with three-layer memory system.

    This agent uses the ContextAssembler to coordinate all memory modules
    and generate contextually relevant responses.
    """

    def __init__(
        self,
        model_name: str = "placeholder",
        context_assembler: Optional[ContextAssembler] = None,
    ) -> None:
        """
        Initialize DM Agent.

        Args:
            model_name: Name of LLM model to use (placeholder for now)
            context_assembler: Custom context assembler (uses singleton if None)
        """
        self.model_name = model_name
        self._assembler = context_assembler or get_context_assembler()

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Process a player query and generate a DM response.

        Args:
            query: Player's question or request
            session_id: Optional session identifier for episodic retrieval

        Returns:
            DM response
        """
        # Step 1: Assemble context from all memory layers
        context = await self._assembler.assemble(
            user_query=query,
            session_id=session_id,
            max_dialogue_turns=10,
        )

        # Step 2: Generate response (placeholder - would call LLM here)
        response = await self._generate_response(context)

        # Step 3: Update dialogue history
        self._assembler.update_dialogue(user_message=query, assistant_message=response)

        return response

    async def process_with_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        force_knowledge: bool = False,
        force_episodes: bool = False,
    ) -> tuple[str, AssembledContext]:
        """
        Process a query and return both response and assembled context.

        Useful for debugging and seeing what context was used.

        Args:
            query: Player's question or request
            session_id: Optional session identifier
            force_knowledge: Force knowledge retrieval
            force_episodes: Force episodic retrieval

        Returns:
            Tuple of (response, assembled_context)
        """
        # Assemble context with overrides
        context = await self._assembler.assemble(
            user_query=query,
            session_id=session_id,
            force_knowledge=force_knowledge,
            force_episodes=force_episodes,
            max_dialogue_turns=10,
        )

        # Generate response
        response = await self._generate_response(context)

        # Update dialogue history
        self._assembler.update_dialogue(user_message=query, assistant_message=response)

        return response, context

    async def _generate_response(self, context: AssembledContext) -> str:
        """
        Generate a response from the LLM.

        Args:
            context: Fully assembled context

        Returns:
            Generated response
        """
        # Format context for LLM
        prompt = self._assembler.format_for_llm(context)

        # Placeholder: In production, integrate with Claude or other LLM
        return (
            "As your Dungeon Master, I understand your question. "
            "I'm using the assembled context to provide you with an accurate response. "
            "In the full implementation, this would be powered by Claude or another LLM."
        )


class DMAgentBuilder:
    """Builder for creating configured DMAgent instances."""

    def __init__(self) -> None:
        """Initialize builder with default configuration."""
        self._model_name: str = "placeholder"
        self._context_assembler: Optional[ContextAssembler] = None

    def with_model(self, model_name: str) -> "DMAgentBuilder":
        """Set the model name."""
        self._model_name = model_name
        return self

    def with_context_assembler(
        self,
        assembler: ContextAssembler,
    ) -> "DMAgentBuilder":
        """Set a custom context assembler."""
        self._context_assembler = assembler
        return self

    def build(self) -> DMAgent:
        """Build and return a DMAgent instance."""
        return DMAgent(
            model_name=self._model_name,
            context_assembler=self._context_assembler,
        )
