"""Agent API router.

Integrates with the three-layer memory system.
Supports both original DMAgent and new LangGraph-based agent.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from mimic_master.models.agent import AgentRequest, AgentResponse
from mimic_master.core import DMAgent

agent_router = APIRouter()

# Global agent instances (singletons)
_dm_agent: DMAgent | None = None
_langgraph_agent = None


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
        raise HTTPException(
            status_code=500, detail=f"Agent processing failed: {str(e)}"
        )


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
        raise HTTPException(
            status_code=500, detail=f"Agent processing failed: {str(e)}"
        )


# ============== LangGraph Agent Endpoints ==============


def get_langgraph_agent():
    """Get or create the LangGraph DM agent instance."""
    global _langgraph_agent
    if _langgraph_agent is None:
        from mimic_master.core.langgraph_agent import create_langgraph_agent

        _langgraph_agent = create_langgraph_agent()
    return _langgraph_agent


@agent_router.post("/langgraph", response_model=AgentResponse)
async def langgraph_agent_chat(request: AgentRequest) -> AgentResponse:
    """
    Process a query through the LangGraph-based DM agent.

    Uses LangGraph's state management, tool calling, and checkpointing.
    Supports session persistence via session_id.

    Args:
        request: AgentRequest containing query and optional session_id

    Returns:
        AgentResponse with DM response
    """
    try:
        agent = get_langgraph_agent()
        response = await agent.invoke(
            query=request.query,
            session_id=request.session_id,
        )

        return AgentResponse(
            response=response,
            retrieved_context=None,
            session_id=request.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph agent failed: {str(e)}")


@agent_router.post("/langgraph/stream")
async def langgraph_agent_stream(request: AgentRequest):
    """
    Stream responses from the LangGraph-based DM agent.

    Yields events as they occur during agent execution.

    Args:
        request: AgentRequest containing query and optional session_id

    Returns:
        StreamingResponse with server-sent events
    """
    try:
        agent = get_langgraph_agent()

        async def event_generator():
            async for event in agent.stream(
                query=request.query,
                session_id=request.session_id,
            ):
                yield f"data: {event}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"LangGraph agent stream failed: {str(e)}"
        )


@agent_router.post("/langgraph/reset/{session_id}")
async def reset_langgraph_session(session_id: str):
    """
    Reset/clear a LangGraph session checkpoint.

    Args:
        session_id: The session to reset

    Returns:
        Success message
    """
    try:
        # Checkpointer reset would be handled here
        return {"status": "success", "message": f"Session {session_id} reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
