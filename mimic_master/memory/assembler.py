"""Context Assembler Module.

The central orchestrator that assembles the final context for the LLM.
Coordinates between all memory modules based on intent.
"""

from typing import Optional, List

from mimic_master.memory.state_memory import StateMemory, get_state_memory
from mimic_master.memory.knowledge_retriever import (
    HybridKnowledgeRetriever,
    get_hybrid_knowledge_retriever,
)
from mimic_master.memory.episodic_retriever import (
    EpisodicRetriever,
    Episode,
    get_episodic_retriever,
)
from mimic_master.memory.intent_classifier import (
    IntentClassifier,
    Intent,
    get_intent_classifier,
)
from mimic_master.models.memory import (
    AssembledContext,
    RetrievalResult,
)

# System prompt template
_SYSTEM_PROMPT_TEMPLATE = """You are an experienced Dungeon Master for Dungeons & Dragons 5th Edition.

## Your Role
- Guide players through an immersive and engaging D&D adventure
- Enforce game rules fairly while maintaining narrative flow
- Be descriptive and evocative when describing scenes
- Balance challenge and fun for the players

## Response Guidelines
- Be helpful and clear about rules when asked
- Use sensory details in descriptions
- Maintain consistency with the story and characters
- Remember important details from previous interactions
- Ask for clarification when needed

## Tone
- Professional yet warm and inviting
- Immersive and atmospheric when describing
- Clear and precise when explaining rules
"""


class ContextAssembler:
    """
    Context assembler that coordinates all memory modules.

    Workflow:
    1. Always retrieve state and dialogue history
    2. Classify user intent
    3. Conditionally trigger knowledge/episodic retrieval
    4. Assemble final context with clear structure
    """

    def __init__(
        self,
        state_memory: Optional[StateMemory] = None,
        knowledge_retriever: Optional[HybridKnowledgeRetriever] = None,
        episodic_retriever: Optional[EpisodicRetriever] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the context assembler.

        Args:
            state_memory: State memory instance (uses singleton if None)
            knowledge_retriever: Knowledge retriever instance
            episodic_retriever: Episodic retriever instance
            intent_classifier: Intent classifier instance
            system_prompt: Custom system prompt (uses default if None)
        """
        self._state_memory = state_memory or get_state_memory()
        self._knowledge_retriever = knowledge_retriever
        self._episodic_retriever = episodic_retriever
        self._intent_classifier = intent_classifier or get_intent_classifier()
        self._system_prompt = system_prompt or _SYSTEM_PROMPT_TEMPLATE

    async def assemble(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        force_knowledge: bool = False,
        force_episodes: bool = False,
        max_dialogue_turns: int = 10,
    ) -> AssembledContext:
        """
        Assemble the full context for the LLM.

        Args:
            user_query: The user's current query
            session_id: Optional session ID for episodic retrieval
            force_knowledge: Force knowledge retrieval regardless of intent
            force_episodes: Force episodic retrieval regardless of intent
            max_dialogue_turns: Number of recent dialogue turns to include

        Returns:
            Fully assembled context ready for the LLM
        """
        # Step 1: Always get state and dialogue history
        state_context = self._state_memory.get_state_context()
        dialogue_context = self._state_memory.get_dialogue_context(max_dialogue_turns)

        # Step 2: Classify intent
        intent = self._intent_classifier.classify(user_query)

        # Step 3: Conditionally retrieve based on intent
        knowledge_context = ""
        episodic_context = ""

        # Knowledge retrieval
        if (
            force_knowledge
            or self._intent_classifier.should_retrieve_rules(intent)
            or self._intent_classifier.should_retrieve_rules(intent)
        ):
            if self._knowledge_retriever:
                knowledge_context = await self._retrieve_knowledge(user_query)

        # Episodic retrieval
        if (
            force_episodes
            or self._intent_classifier.should_retrieve_episodes(intent)
        ):
            if self._episodic_retriever:
                episodic_context = await self._retrieve_episodes(user_query, session_id)

        # Step 4: Build system prompt
        system_prompt = self._build_system_prompt(intent)

        # Step 5: Assemble final context
        return AssembledContext(
            system_prompt=system_prompt,
            state_context=state_context,
            dialogue_history=dialogue_context,
            knowledge_context=knowledge_context,
            episodic_context=episodic_context,
            user_query=user_query,
        )

    def _build_system_prompt(self, intent: Intent) -> str:
        """
        Build the system prompt with intent-specific adjustments.

        Args:
            intent: Classified intent

        Returns:
            System prompt string
        """
        base_prompt = self._system_prompt

        # Add intent-specific instructions
        intent_guidance = self._get_intent_guidance(intent)

        if intent_guidance:
            return f"{base_prompt}\n\n{intent_guidance}"

        return base_prompt

    def _get_intent_guidance(self, intent: Intent) -> str:
        """
        Get intent-specific guidance for the system prompt.

        Args:
            intent: Classified intent

        Returns:
            Guidance string
        """
        from mimic_master.memory.intent_classifier import IntentType

        try:
            intent_type = IntentType(intent.type)
        except ValueError:
            return ""

        guidance_map = {
            IntentType.QUERY_RULES: """## Current Focus: Rules Query
The user is asking about game rules. Be precise and cite sources when possible.
Provide clear explanations with examples. If you're retrieving rules, use the provided context.""",
            IntentType.COMBAT: """## Current Focus: Combat
The user is involved in combat. Be quick and decisive. Track initiative, HP, and conditions.
Prioritize action economy and clear rulings.""",
            IntentType.PROCEED_STORY: """## Current Focus: Story Progression
The user is moving the story forward. Be descriptive and engaging.
Introduce interesting plot developments and maintain narrative momentum.""",
            IntentType.RECALL_HISTORY: """## Current Focus: Recall
The user is asking about past events. Use the retrieved episode summaries to help.
Be accurate about what happened before while keeping the conversation natural.""",
            IntentType.CHAT: """## Current Focus: Conversation
The user is chatting casually. Be friendly and engaging.
Stay in character as the Dungeon Master but keep things light.""",
        }

        return guidance_map.get(intent_type, "")

    async def _retrieve_knowledge(self, query: str) -> str:
        """
        Retrieve relevant rules/knowledge and format as context.

        Args:
            query: Query text

        Returns:
            Formatted knowledge context
        """
        try:
            results = await self._knowledge_retriever.retrieve(query, top_k=5)

            if not results:
                return ""

            lines = ["## Relevant Rules & Information\n"]
            for i, result in enumerate(results, 1):
                content = result.content[:500] + "..." if len(result.content) > 500 else result.content
                lines.append(f"### Source {i}")
                lines.append(f"{content}")
                if result.metadata:
                    source = result.metadata.get("source", "Unknown")
                    lines.append(f"*Source: {source}*\n")

            return "\n".join(lines)

        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return ""

    async def _retrieve_episodes(
        self,
        query: str,
        session_id: Optional[str],
    ) -> str:
        """
        Retrieve relevant episodes and format as context.

        Args:
            query: Query text
            session_id: Optional session ID filter

        Returns:
            Formatted episodic context
        """
        try:
            episodes = await self._episodic_retriever.retrieve(
                query=query,
                session_id=session_id,
                top_k=3,
            )

            if not episodes:
                return ""

            lines = ["## Relevant Past Events\n"]
            for i, episode in enumerate(episodes, 1):
                lines.append(f"### Episode {i}")
                lines.append(f"{episode.summary}")
                if episode.key_events:
                    lines.append("**Key Events:**")
                    for event in episode.key_events:
                        lines.append(f"- {event}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            print(f"Episodic retrieval error: {e}")
            return ""

    def format_for_llm(self, context: AssembledContext) -> str:
        """
        Format the assembled context as a single string for the LLM.

        Args:
            context: Assembled context

        Returns:
            Formatted string ready for LLM input
        """
        parts = [
            context.system_prompt,
            context.state_context,
            context.dialogue_history,
        ]

        if context.knowledge_context:
            parts.append(context.knowledge_context)

        if context.episodic_context:
            parts.append(context.episodic_context)

        parts.append(f"## Current Query\n{context.user_query}")

        return "\n\n".join(parts)

    def update_dialogue(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Update dialogue history with a turn.

        Args:
            user_message: User's message
            assistant_message: Assistant's response
        """
        self._state_memory.add_dialogue("user", user_message)
        self._state_memory.add_dialogue("assistant", assistant_message)


# Singleton instance
_context_assembler: Optional[ContextAssembler] = None


def get_context_assembler(
    state_memory: Optional[StateMemory] = None,
    knowledge_retriever: Optional[HybridKnowledgeRetriever] = None,
    episodic_retriever: Optional[EpisodicRetriever] = None,
    intent_classifier: Optional[IntentClassifier] = None,
    system_prompt: Optional[str] = None,
) -> ContextAssembler:
    """Get the singleton context assembler instance."""
    global _context_assembler
    if _context_assembler is None:
        _context_assembler = ContextAssembler(
            state_memory=state_memory,
            knowledge_retriever=knowledge_retriever,
            episodic_retriever=episodic_retriever,
            intent_classifier=intent_classifier,
            system_prompt=system_prompt,
        )
    return _context_assembler


def reset_context_assembler() -> None:
    """Reset the singleton context assembler instance."""
    global _context_assembler
    _context_assembler = None
