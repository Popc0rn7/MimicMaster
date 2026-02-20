"""Agent API router."""

from fastapi import APIRouter, HTTPException
from typing import List

from mimic_master.models.agent import AgentRequest, AgentResponse
from mimic_master.models.retrieval import RetrievalRequest

from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.services.reranker_service import get_reranker_service

agent_router = APIRouter()


@agent_router.post("/", response_model=AgentResponse)
async def agent_chat(request: AgentRequest) -> AgentResponse:
    """
    Process a query through the DM agent with RAG retrieval.

    Args:
        request: AgentRequest containing query, optional context, and session_id

    Returns:
        AgentResponse with DM response and retrieved context
    """
    try:
        # Step 1: Retrieve relevant documents
        embedding_service = get_embedding_service()
        embedding_response = await embedding_service.embed([request.query])
        query_embedding = embedding_response.embeddings[0]

        pinecone_service = get_pinecone_service()
        retrieved_docs = await pinecone_service.query(
            query_embedding=query_embedding,
            top_k=10,
            namespace="",  # Can be configured based on campaign/world
        )

        # Step 2: Rerank if we have multiple results
        retrieved_context: List[str] = []
        if len(retrieved_docs) > 0:
            if len(retrieved_docs) > 3:
                # Use reranker for more results
                reranker_service = get_reranker_service()
                doc_texts = [doc.content for doc in retrieved_docs]
                rerank_response = await reranker_service.rerank(
                    query=request.query,
                    documents=doc_texts,
                    top_n=5,
                )
                retrieved_context = [
                    retrieved_docs[idx].content for idx in rerank_response.results
                ]
            else:
                retrieved_context = [doc.content for doc in retrieved_docs]

        # Step 3: Merge context if provided
        if request.context:
            retrieved_context = request.context + retrieved_context

        # Step 4: Generate DM response
        # Note: This is a placeholder - actual LLM integration would go here
        dm_response = await _generate_dm_response(request.query, retrieved_context)

        return AgentResponse(
            response=dm_response,
            retrieved_context=retrieved_context[:10],  # Limit context size
            session_id=request.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


async def _generate_dm_response(query: str, context: List[str]) -> str:
    """
    Generate a DM response based on query and context.

    Note: This is a placeholder implementation.
    In production, this would call an LLM (e.g., Claude via Anthropic API).

    Args:
        query: The player's query
        context: Retrieved relevant context

    Returns:
        DM response
    """
    # Placeholder: simple template response
    if context:
        context_str = "\n".join([f"- {c[:100]}..." for c in context[:3]])
        return (
            f"As your Dungeon Master, I have found some relevant information:\n\n"
            f"{context_str}\n\n"
            f"Regarding your query about '{query}', let me think about that..."
        )
    return (
        f"Interesting question about '{query}'. "
        f"As your Dungeon Master, I'll consider the game mechanics and lore..."
    )
