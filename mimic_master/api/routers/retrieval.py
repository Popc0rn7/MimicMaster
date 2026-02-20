"""Retrieval API router."""

from fastapi import APIRouter, HTTPException

from mimic_master.models.retrieval import RetrievalRequest, RetrievalResponse
from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.embedding_service import get_embedding_service

retrieval_router = APIRouter()


@retrieval_router.post("/", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """
    Retrieve relevant documents from the vector database.

    Args:
        request: RetrievalRequest containing query and retrieval parameters

    Returns:
        RetrievalResponse with retrieved documents
    """
    try:
        # Generate embedding for the query
        embedding_service = get_embedding_service()
        embedding_response = await embedding_service.embed([request.query])
        query_embedding = embedding_response.embeddings[0]

        # Query Pinecone
        pinecone_service = get_pinecone_service()
        results = await pinecone_service.query(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filter_dict=request.filter,
            namespace=request.namespace,
        )

        return RetrievalResponse(
            results=results,
            total=len(results),
            query=request.query,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@retrieval_router.post("/upsert")
async def upsert_documents(
    ids: list[str],
    texts: list[str],
    namespace: str = "",
    metadata: list[dict] | None = None,
) -> dict:
    """
    Upsert documents into the vector database.

    Args:
        ids: List of document IDs
        texts: List of document contents
        namespace: Namespace for the documents
        metadata: Optional list of metadata dictionaries

    Returns:
        Confirmation message
    """
    try:
        pinecone_service = get_pinecone_service()
        await pinecone_service.upsert_from_texts(
            ids=ids,
            texts=texts,
            metadata=metadata,
            namespace=namespace,
        )
        return {"message": f"Successfully upserted {len(ids)} documents"}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")
