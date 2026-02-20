"""Agent request and response models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request model for DM agent."""

    query: str = Field(..., description="Player query or context")
    context: Optional[List[str]] = Field(
        default=None, description="Additional context (retrieved documents)"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class AgentResponse(BaseModel):
    """Response model from DM agent."""

    response: str = Field(..., description="DM response")
    retrieved_context: Optional[List[str]] = Field(
        default=None, description="Context used for generating response"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")
