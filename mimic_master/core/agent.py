"""Core DM Agent implementation."""

from typing import List, Optional, Dict, Any


class DMAgent:
    """
    Dungeon Master Agent for D&D 5E.

    This agent handles player queries and generates contextually relevant responses
    using retrieved information from the vector database.
    """

    def __init__(self, model_name: str = "placeholder") -> None:
        """
        Initialize the DM Agent.

        Args:
            model_name: Name of the LLM model to use (placeholder for now)
        """
        self.model_name = model_name
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the DM Agent.

        Returns:
            System prompt string
        """
        return """You are an experienced Dungeon Master for Dungeons & Dragons 5th Edition.
Your role is to help players understand game rules, lore, and mechanics.

When responding:
- Be helpful and clear
- Reference specific rules and sources when possible
- Use examples to illustrate complex concepts
- Consider the context of the player's situation
- Maintain an engaging, immersive tone

If you don't have enough information to answer a question, ask for clarification.
"""

    async def process_query(
        self,
        query: str,
        context: List[str],
        session_id: Optional[str] = None,
    ) -> str:
        """
        Process a player query and generate a DM response.

        Args:
            query: Player's question or request
            context: Retrieved relevant documents
            session_id: Optional session identifier for conversation memory

        Returns:
            DM response
        """
        # Note: This is a placeholder implementation
        # In production, this would call an LLM (e.g., Claude via Anthropic API)

        context_str = self._format_context(context)
        prompt = self._build_prompt(query, context_str)

        # Placeholder response generation
        return await self._generate_response(prompt)

    def _format_context(self, context: List[str]) -> str:
        """
        Format retrieved context for the prompt.

        Args:
            context: List of context strings

        Returns:
            Formatted context string
        """
        if not context:
            return "No relevant information found."

        formatted = []
        for i, ctx in enumerate(context, 1):
            # Truncate very long context entries
            ctx_text = ctx[:500] + "..." if len(ctx) > 500 else ctx
            formatted.append(f"[{i}] {ctx_text}")

        return "\n\n".join(formatted)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the full prompt for the LLM.

        Args:
            query: Player's query
            context: Formatted context

        Returns:
            Full prompt
        """
        return f"""{self._system_prompt}

Relevant Information:
{context}

Player Query: {query}

Your Response:"""

    async def _generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Full prompt to send to the LLM

        Returns:
            Generated response
        """
        # Placeholder: In production, integrate with Claude or other LLM
        return (
            "As your Dungeon Master, I understand your question. "
            "I'm using the context I retrieved to provide you with an accurate response. "
            "In the full implementation, this would be powered by Claude or another LLM."
        )


class DMAgentBuilder:
    """Builder for creating configured DMAgent instances."""

    def __init__(self) -> None:
        """Initialize the builder with default configuration."""
        self._model_name: str = "placeholder"
        self._system_prompt: Optional[str] = None
        self._temperature: float = 0.7
        self._max_tokens: int = 1000

    def with_model(self, model_name: str) -> "DMAgentBuilder":
        """Set the model name."""
        self._model_name = model_name
        return self

    def with_system_prompt(self, prompt: str) -> "DMAgentBuilder":
        """Set a custom system prompt."""
        self._system_prompt = prompt
        return self

    def with_temperature(self, temperature: float) -> "DMAgentBuilder":
        """Set the generation temperature."""
        self._temperature = temperature
        return self

    def with_max_tokens(self, max_tokens: int) -> "DMAgentBuilder":
        """Set the maximum tokens for generation."""
        self._max_tokens = max_tokens
        return self

    def build(self) -> DMAgent:
        """Build and return a DMAgent instance."""
        agent = DMAgent(model_name=self._model_name)
        if self._system_prompt:
            agent._system_prompt = self._system_prompt
        return agent
