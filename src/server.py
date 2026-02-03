from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedding_text import ChunkingConfig, TextEmbedder, embed_markdown


app = FastAPI(title="embedding-service", version="0.1.0")


class EmbedRequest(BaseModel):
	markdown: str = Field(..., description="Markdown text to embed")
	max_tokens: int = Field(512, ge=16, le=4096)
	overlap_tokens: int = Field(64, ge=0, le=1024)
	min_chunk_chars: int = Field(20, ge=0, le=10_000)
	include_chunks: bool = Field(True, description="Whether to return text chunks")


class EmbedResponse(BaseModel):
	model: str
	chunks: list[str] | None
	vectors: list[list[float]]


_embedder: TextEmbedder | None = None


def get_embedder() -> TextEmbedder:
	global _embedder
	if _embedder is None:
		_embedder = TextEmbedder(
			model_name="BAAI/bge-base-zh-v1.5",
			device="auto",
			batch_size=16,
			normalize=True,
		)
	return _embedder


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
	if not req.markdown.strip():
		raise HTTPException(status_code=400, detail="markdown is empty")

	chunking = ChunkingConfig(
		max_tokens=req.max_tokens,
		overlap_tokens=req.overlap_tokens,
		min_chunk_chars=req.min_chunk_chars,
	)
	embedder = get_embedder()
	result = embed_markdown(req.markdown, embedder=embedder, chunking=chunking)
	return EmbedResponse(
		model=result["model"],
		chunks=result["chunks"] if req.include_chunks else None,
		vectors=result["vectors"],
	)


class EmbedFileRequest(BaseModel):
	path: str = Field(..., description="Relative path under inputs/, e.g. 玩家手册（2024）.md")
	max_tokens: int = Field(512, ge=16, le=4096)
	overlap_tokens: int = Field(64, ge=0, le=1024)
	min_chunk_chars: int = Field(20, ge=0, le=10_000)
	include_chunks: bool = Field(True)


@app.post("/embed-file", response_model=EmbedResponse)
def embed_file(req: EmbedFileRequest) -> EmbedResponse:
	inputs_root = (Path(__file__).resolve().parent.parent / "inputs").resolve()
	target = (inputs_root / req.path).resolve()

	if inputs_root not in target.parents and target != inputs_root:
		raise HTTPException(status_code=400, detail="path must be under inputs/")
	if not target.exists() or not target.is_file():
		raise HTTPException(status_code=404, detail="file not found")
	if target.suffix.lower() != ".md":
		raise HTTPException(status_code=400, detail="only .md files are supported")

	markdown = target.read_text(encoding="utf-8", errors="ignore")
	chunking = ChunkingConfig(
		max_tokens=req.max_tokens,
		overlap_tokens=req.overlap_tokens,
		min_chunk_chars=req.min_chunk_chars,
	)
	embedder = get_embedder()
	result = embed_markdown(markdown, embedder=embedder, chunking=chunking)
	return EmbedResponse(
		model=result["model"],
		chunks=result["chunks"] if req.include_chunks else None,
		vectors=result["vectors"],
	)
