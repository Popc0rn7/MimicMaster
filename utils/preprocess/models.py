"""Pydantic models for normalized knowledge records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ContentKind = Literal["narrative", "structured_entity", "table"]


class SourceTrace(BaseModel):
    """Traceability metadata for a normalized record."""

    provider: str
    source_file: str
    source_key: str
    source_index: int | None = None
    source_commit: str
    license: str


class NormalizedContent(BaseModel):
    """Normalized record payload."""

    kind: ContentKind
    entries: list[Any] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    rendered_text: str


class NormalizedRecord(BaseModel):
    """A single line in knowledge/normalized/*.jsonl."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_book: Literal["PHB", "DMG", "MM"]
    record_type: str
    title: str
    eng_title: str | None = None
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    source_trace: SourceTrace
    content: NormalizedContent


class BuildManifest(BaseModel):
    """Summary of a normalization run."""

    generated_at: str
    source_commit: str
    input_root: str
    output_root: str
    sources: list[str] = Field(default_factory=list)
    outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    skipped_items: int = 0
    unknown_tags: dict[str, int] = Field(default_factory=dict)
    empty_records: int = 0
    source_errors: list[str] = Field(default_factory=list)
