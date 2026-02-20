"""Intent Classifier Module.

Classifies user queries into intent types to determine which retrieval
modules to activate.

Intents:
- QUERY_RULES: Questions about game rules, mechanics, spells, etc.
- PROCEED_STORY: Moving the plot forward, describing scenes
- COMBAT: Combat-related actions and rulings
- RECALL_HISTORY: Remembering past events, asking about previous sessions
- CHAT: General conversation, roleplay, etc.
"""

from enum import Enum
from typing import List, Dict, Set, Optional
import re

from mimic_master.models.memory import Intent


class IntentType(str, Enum):
    """Intent types for DM Agent queries."""

    QUERY_RULES = "query_rules"  # Rules, mechanics, spells, etc.
    PROCEED_STORY = "proceed_story"  # Moving plot, describing scenes
    COMBAT = "combat"  # Combat actions, rulings
    RECALL_HISTORY = "recall_history"  # Remembering past events
    CHAT = "chat"  # General conversation, roleplay


# Keyword patterns for each intent
_INTENT_PATTERNS: Dict[IntentType, List[Dict[str, str]]] = {
    IntentType.QUERY_RULES: [
        {"keywords": "rule,rules,how do,how to,what is,what are", "weight": 1.0},
        {"keywords": "spell,casting,caster,slot,slot,attack,damage,save,ability,score", "weight": 0.8},
        {"keywords": "advantage,disadvantage,proficiency,modifier,bonus,penalty", "weight": 0.8},
        {"keywords": "class,subclass,level,feat,skill,tool,weapon,armor", "weight": 0.7},
        {"keywords": "condition,status,effect,duration,concentration", "weight": 0.7},
        {"keywords": "dc,difficulty,check,roll,throw,dice", "weight": 0.6},
    ],
    IntentType.PROCEED_STORY: [
        {"keywords": "what happens next,then what,where do we go,continue,proceed", "weight": 1.0},
        {"keywords": "describe,look around,investigate,search,examine", "weight": 0.8},
        {"keywords": "enter,go to,move,towards,approach,head", "weight": 0.7},
        {"keywords": "rest,take a break,long rest,short rest", "weight": 0.6},
    ],
    IntentType.COMBAT: [
        {"keywords": "attack,hit,strike,swing,shoot,cast,fire", "weight": 1.0},
        {"keywords": "defend,block,parry,dodge,dodge action", "weight": 0.8},
        {"keywords": "initiative,turn,round,combat,enemy,foe,monster", "weight": 0.9},
        {"keywords": "damage,deal,take,heal,restore,cure", "weight": 0.7},
        {"keywords": "action,bonus action,reaction,move", "weight": 0.6},
    ],
    IntentType.RECALL_HISTORY: [
        {"keywords": "remember,recall,what did we,what happened,last session,before", "weight": 1.0},
        {"keywords": "previous,past,earlier,ago", "weight": 0.8},
        {"keywords": "met,found,discovered,learned", "weight": 0.6},
    ],
    IntentType.CHAT: [
        {"keywords": "hello,hi,hey,greetings", "weight": 1.0},
        {"keywords": "thanks,thank you,awesome,cool,great", "weight": 0.8},
        {"keywords": "joke,laugh,funny,humor", "weight": 0.7},
        {"keywords": "feel,think,believe,opinion", "weight": 0.5},
    ],
}


class IntentClassifier:
    """
    Intent classifier for determining which memory modules to activate.

    Uses rule-based keyword matching for quick classification.
    Can be extended with LLM-based classification for complex queries.
    """

    def __init__(self, use_llm_fallback: bool = False) -> None:
        """
        Initialize the intent classifier.

        Args:
            use_llm_fallback: Whether to use LLM for uncertain classifications
        """
        self._use_llm_fallback = use_llm_fallback
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for each intent."""
        self._compiled_patterns: Dict[IntentType, List[tuple[str, float]]] = {}

        for intent_type, patterns in _INTENT_PATTERNS.items():
            compiled = []
            for pattern in patterns:
                keywords = pattern["keywords"].split(",")
                # Build a regex pattern that matches any of the keywords
                regex_pattern = r"\b(" + "|".join(re.escape(k.strip()) for k in keywords) + r")\b"
                compiled.append((regex_pattern, pattern["weight"]))
            self._compiled_patterns[intent_type] = compiled

    def classify(self, query: str) -> Intent:
        """
        Classify the intent of a user query.

        Args:
            query: User query text

        Returns:
            Intent classification
        """
        query_lower = query.lower()

        # Score each intent
        scores: Dict[IntentType, float] = {}

        for intent_type, patterns in self._compiled_patterns.items():
            score = 0.0
            for pattern, weight in patterns:
                matches = re.findall(pattern, query_lower)
                score += len(matches) * weight

            # Normalize by query length to prevent long queries from having advantage
            if len(query_lower) > 0:
                score = score / len(query_lower.split())

            scores[intent_type] = score

        # Find the highest scoring intent
        best_intent = max(scores.items(), key=lambda x: x[1])

        # Default to CHAT if no clear intent
        if best_intent[1] < 0.05:
            best_intent = (IntentType.CHAT, 0.0)

        # Handle tie-breaker for similar scores
        close_intents = [
            intent for intent, score in scores.items()
            if abs(score - best_intent[1]) < 0.01 and score > 0.05
        ]

        # If we have a tie, use fallback logic
        if len(close_intents) > 1:
            if self._use_llm_fallback:
                return self._classify_with_llm(query, close_intents)
            else:
                # Default order: QUERY_RULES > COMBAT > PROCEED_STORY > RECALL_HISTORY > CHAT
                priority_order = [
                    IntentType.QUERY_RULES,
                    IntentType.COMBAT,
                    IntentType.PROCEED_STORY,
                    IntentType.RECALL_HISTORY,
                    IntentType.CHAT,
                ]
                for intent in priority_order:
                    if intent in close_intents:
                        best_intent = (intent, scores[intent])
                        break

        return Intent(type=best_intent[0].value, confidence=min(best_intent[1], 1.0))

    def _classify_with_llm(
        self,
        query: str,
        candidates: List[IntentType],
    ) -> Intent:
        """
        Classify with LLM for uncertain cases (placeholder).

        Args:
            query: User query
            candidates: Candidate intents

        Returns:
            Intent classification
        """
        # Placeholder: In production, this would call an LLM
        # For now, return the first candidate
        return Intent(type=candidates[0].value, confidence=0.7)

    def get_intent_description(self, intent_type: IntentType) -> str:
        """
        Get a human-readable description of an intent type.

        Args:
            intent_type: Intent type

        Returns:
            Description string
        """
        descriptions = {
            IntentType.QUERY_RULES: "Querying rules, mechanics, spells, or game information",
            IntentType.PROCEED_STORY: "Moving the story forward or exploring",
            IntentType.COMBAT: "Combat-related actions and rulings",
            IntentType.RECALL_HISTORY: "Remembering past events or sessions",
            IntentType.CHAT: "General conversation or roleplay",
        }
        return descriptions.get(intent_type, "Unknown intent")

    def should_retrieve_rules(self, intent: Intent) -> bool:
        """Check if rules retrieval is needed for this intent."""
        return intent.type in [
            IntentType.QUERY_RULES.value,
            IntentType.COMBAT.value,
        ]

    def should_retrieve_episodes(self, intent: Intent) -> bool:
        """Check if episodic retrieval is needed for this intent."""
        return intent.type == IntentType.RECALL_HISTORY.value

    def should_skip_rag(self, intent: Intent) -> bool:
        """Check if RAG should be skipped for this intent."""
        return intent.type == IntentType.CHAT.value


# Singleton instance
_intent_classifier: Optional[IntentClassifier] = None


def get_intent_classifier(use_llm_fallback: bool = False) -> IntentClassifier:
    """Get the singleton intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier(use_llm_fallback)
    return _intent_classifier
