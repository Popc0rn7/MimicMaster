"""Agent API router.

Integrates with the three-layer memory system.
"""

from fastapi import APIRouter, HTTPException

from mimic_master.models.agent import AgentRequest, AgentResponse
from mimic_master.core import DMAgent

agent_router = APIRouter()

# Global agent instance (singleton)
_dm_agent: DMAgent | None = None


def get_dm_agent() -> DMAgent:
    """Get or create the DM agent instance."""
    global _dm_agent
    if _dm_agent is None:
        _dm_agent = DMAgent()
    return _dm_agent


@agent_router.post("/", response_model=AgentResponse)
async def agent_chat(request: AgentRequest) -> AgentResponse:
    """
    Process a query through the DM agent with three-layer memory system.

    The agent will:
    1. Classify the user's intent
    2. Retrieve current state and dialogue history
    3. Conditionally retrieve rules/knowledge (if relevant)
    4. Conditionally retrieve episodic memories (if relevant)
    5. Generate a context-aware response

    Args:
        request: AgentRequest containing query and optional session_id

    Returns:
        AgentResponse with DM response
    """
    try:
        agent = get_dm_agent()
        response = await agent.process_query(
            query=request.query,
            session_id=request.session_id,
        )

        return AgentResponse(
            response=response,
            retrieved_context=None,  # Context is now managed internally
            session_id=request.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


@agent_router.post("/with-context")
async def agent_chat_with_context(request: AgentRequest) -> AgentResponse:
    """
    Process a query and return the full assembled context (debug mode).

    Useful for debugging to see what context was assembled.

    Args:
        request: AgentRequest containing query and optional session_id

    Returns:
        AgentResponse with DM response and context details
    """
    try:
        agent = get_dm_agent()
        response, context = await agent.process_with_context(
            query=request.query,
            session_id=request.session_id,
        )

        # Return the assembled context for debugging
        return AgentResponse(
            response=response,
            retrieved_context=[context.model_dump_json()],
            session_id=request.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")
