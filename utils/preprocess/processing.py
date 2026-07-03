"""Process normalized records into MongoDB and Pinecone import shapes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from utils.preprocess.models import BuildManifest, NormalizedRecord, SourceTrace
from utils.preprocess.writers import new_timestamp, sha256_file, write_json

RetrievalMode = Literal["mongo_exact", "pinecone_semantic", "both_fallback"]

MONGO_EXACT_TYPES = {
    "background",
    "class",
    "class_feature",
    "condition",
    "disease",
    "equipment",
    "feat",
    "hazard",
    "item",
    "item_group",
    "magic_item",
    "monster",
    "object",
    "optional_feature",
    "race",
    "reward",
    "spell",
    "status",
    "subclass",
    "subclass_feature",
    "subrace",
    "table",
    "trap",
}
PINECONE_SEMANTIC_TYPES = {
    "dm_section",
    "monster_fluff",
    "monster_section",
    "rule_section",
    "variant_rule",
}
DAMAGE_KEYS = {
    "acid": "damage:acid",
    "cold": "damage:cold",
    "fire": "damage:fire",
    "force": "damage:force",
    "lightning": "damage:lightning",
    "necrotic": "damage:necrotic",
    "poison": "damage:poison",
    "psychic": "damage:psychic",
    "radiant": "damage:radiant",
    "thunder": "damage:thunder",
}
TEXT_TAGS = {
    "掩护": "combat",
    "遭遇": "encounter",
    "魔法物品": "magic_item",
    "休息": "rest",
}


class MongoEntity(BaseModel):
    """A MongoDB source-of-truth document for exact lookup."""

    id: str = Field(alias="_id")
    retrieval_mode: RetrievalMode
    source_book: str
    record_type: str
    content_kind: str
    title: str
    eng_title: str | None = None
    aliases: list[str] = Field(default_factory=list)
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    record_tags: list[str] = Field(default_factory=list)
    mechanic_tags: list[str] = Field(default_factory=list)
    numeric_filters: dict[str, Any] = Field(default_factory=dict)
    structured: dict[str, Any] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    rendered_text: str
    source_trace: SourceTrace
    normalized_id: str


class PineconeChunk(BaseModel):
    """A Pinecone vector payload that can answer without Mongo hydration."""

    id: str = Field(alias="_id")
    retrieval_mode: RetrievalMode
    source_book: str
    record_type: str
    content_kind: str
    chunk_kind: str
    title: str
    eng_title: str | None = None
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    semantic_tags: list[str] = Field(default_factory=list)
    mechanic_tags: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    numeric_filters: dict[str, Any] = Field(default_factory=dict)
    source_trace: SourceTrace
    normalized_id: str
    chunk_index: int


class ProcessedOutputs(BaseModel):
    """In-memory processing result."""

    mongo_entities: list[MongoEntity] = Field(default_factory=list)
    pinecone_chunks: list[PineconeChunk] = Field(default_factory=list)


@dataclass
class ProcessStats:
    routed_to_mongo: int = 0
    routed_to_pinecone: int = 0
    skipped_empty: int = 0
    both_fallback: int = 0
    unknown_record_types: dict[str, int] = field(default_factory=dict)


def process_records(
    records: list[NormalizedRecord], stats: ProcessStats | None = None
) -> ProcessedOutputs:
    """Split normalized records into exact Mongo entities and semantic chunks."""

    stats = stats or ProcessStats()
    outputs = ProcessedOutputs()
    for record in records:
        text = record.content.rendered_text.strip()
        if not text:
            stats.skipped_empty += 1
            continue
        if should_route_to_mongo(record):
            outputs.mongo_entities.append(to_mongo_entity(record))
            stats.routed_to_mongo += 1
        elif should_route_to_pinecone(record):
            chunks = to_pinecone_chunks(record)
            outputs.pinecone_chunks.extend(chunks)
            stats.routed_to_pinecone += len(chunks)
        else:
            stats.unknown_record_types[record.record_type] = (
                stats.unknown_record_types.get(record.record_type, 0) + 1
            )
            outputs.pinecone_chunks.extend(to_pinecone_chunks(record))
            stats.routed_to_pinecone += 1
            stats.both_fallback += 1
    return outputs


def process_to_outputs(
    *, normalized_dir: Path, output_root: Path, limit: int | None = None
) -> BuildManifest:
    """Read normalized JSONL files and write Mongo/Pinecone import JSONL."""

    records = read_normalized_records(normalized_dir, limit=limit)
    stats = ProcessStats()
    processed = process_records(records, stats=stats)
    mongo_path = output_root / "mongo" / "entities.jsonl"
    pinecone_path = output_root / "pinecone" / "chunks.jsonl"
    mongo_meta = write_processed_jsonl(mongo_path, processed.mongo_entities)
    pinecone_meta = write_processed_jsonl(pinecone_path, processed.pinecone_chunks)
    manifest = BuildManifest(
        generated_at=new_timestamp(),
        source_commit=_source_commit(records),
        input_root=str(normalized_dir),
        output_root=str(output_root),
        sources=["process"],
        outputs={
            "mongo_entities": mongo_meta,
            "pinecone_chunks": pinecone_meta,
        },
        skipped_items=stats.skipped_empty,
        unknown_tags=stats.unknown_record_types,
        empty_records=stats.skipped_empty,
        source_errors=[],
    )
    write_json(
        output_root / "manifests" / "processing-latest.json",
        {
            **manifest.model_dump(mode="json"),
            "routing": {
                "mongo_exact": stats.routed_to_mongo,
                "pinecone_semantic": stats.routed_to_pinecone,
                "both_fallback": stats.both_fallback,
            },
        },
    )
    write_processing_manifest(output_root, manifest, stats)
    return manifest


def read_normalized_records(
    normalized_dir: Path, limit: int | None = None
) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    for path in sorted(normalized_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if limit is not None and len(records) >= limit:
                    return records
                records.append(NormalizedRecord.model_validate_json(line))
    return records


def should_route_to_mongo(record: NormalizedRecord) -> bool:
    return record.content.kind == "table" or record.record_type in MONGO_EXACT_TYPES


def should_route_to_pinecone(record: NormalizedRecord) -> bool:
    return (
        record.record_type in PINECONE_SEMANTIC_TYPES
        or record.content.kind == "narrative"
    )


def to_mongo_entity(record: NormalizedRecord) -> MongoEntity:
    structured = record.content.structured
    return MongoEntity(
        _id=record.id,
        retrieval_mode="mongo_exact",
        source_book=record.source_book,
        record_type=record.record_type,
        content_kind=record.content.kind,
        title=record.title,
        eng_title=record.eng_title,
        aliases=aliases_for(record),
        page=record.page,
        section_path=record.section_path,
        record_tags=record_tags_for(record),
        mechanic_tags=mechanic_tags_for(record),
        numeric_filters=numeric_filters_for(record),
        structured=structured,
        tables=record.content.tables,
        rendered_text=record.content.rendered_text,
        source_trace=record.source_trace,
        normalized_id=record.id,
    )


def to_pinecone_chunks(record: NormalizedRecord) -> list[PineconeChunk]:
    chunks = split_text(record.content.rendered_text)
    return [
        PineconeChunk(
            _id=f"pinecone:{record.id}:{index}",
            retrieval_mode="pinecone_semantic",
            source_book=record.source_book,
            record_type=record.record_type,
            content_kind=record.content.kind,
            chunk_kind=chunk_kind_for(record),
            title=record.title,
            eng_title=record.eng_title,
            page=record.page,
            section_path=record.section_path,
            text=text,
            semantic_tags=semantic_tags_for(record, text),
            mechanic_tags=mechanic_tags_for(record),
            entity_refs=entity_refs_for(record),
            numeric_filters=numeric_filters_for(record),
            source_trace=record.source_trace,
            normalized_id=record.id,
            chunk_index=index,
        )
        for index, text in enumerate(chunks)
    ]


def aliases_for(record: NormalizedRecord) -> list[str]:
    aliases = []
    for value in (record.title, record.eng_title):
        if value and value not in aliases:
            aliases.append(value)
    return aliases


def record_tags_for(record: NormalizedRecord) -> list[str]:
    tags = {record.record_type, record.content.kind, record.source_book.lower()}
    if record.record_type == "table":
        tags.add("table")
    return sorted(tags)


def mechanic_tags_for(record: NormalizedRecord) -> list[str]:
    tags: set[str] = set()
    structured = record.content.structured
    for damage in structured.get("damageInflict", []) or []:
        if isinstance(damage, str):
            tags.add(DAMAGE_KEYS.get(damage, f"damage:{damage}"))
    for saving_throw in structured.get("savingThrow", []) or []:
        if isinstance(saving_throw, str):
            tags.add(f"saving_throw:{saving_throw}")
    if structured.get("duration"):
        if "concentration" in str(structured.get("duration")).lower():
            tags.add("concentration")
    return sorted(tags)


def numeric_filters_for(record: NormalizedRecord) -> dict[str, Any]:
    structured = record.content.structured
    filters: dict[str, Any] = {}
    if record.page is not None:
        filters["page"] = record.page
    if "level" in structured:
        filters["spell_level"] = structured["level"]
    if "cr" in structured:
        filters["cr"] = structured["cr"]
    if "ac" in structured:
        filters["ac"] = structured["ac"]
    return filters


def semantic_tags_for(record: NormalizedRecord, text: str) -> list[str]:
    tags = {record.record_type, record.source_book.lower()}
    if record.content.kind == "narrative":
        tags.add("rule_text")
    if record.record_type == "monster_fluff":
        tags.add("lore")
    searchable_context = " ".join(
        [record.title, record.eng_title or "", *record.section_path]
    )
    for phrase, tag in TEXT_TAGS.items():
        if phrase in searchable_context or phrase in text[:120]:
            tags.add(tag)
    return sorted(tags)


def entity_refs_for(record: NormalizedRecord) -> list[str]:
    refs = []
    for value in (record.title, record.eng_title):
        if value and value not in refs:
            refs.append(value)
    return refs


def chunk_kind_for(record: NormalizedRecord) -> str:
    if record.record_type == "monster_fluff":
        return "flavor"
    if record.record_type in {
        "rule_section",
        "dm_section",
        "monster_section",
        "variant_rule",
    }:
        return "rule_text"
    return "semantic_text"


def split_text(text: str, max_chars: int = 1200) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + 1 + len(paragraph) <= max_chars:
            current = f"{current}\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def write_processed_jsonl(path: Path, records: list[BaseModel]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record.model_dump(by_alias=True, mode="json"), ensure_ascii=False
                )
                + "\n"
            )
    return {
        "path": str(path),
        "record_count": len(records),
        "sha256": sha256_file(path),
    }


def write_processing_manifest(
    output_root: Path, manifest: BuildManifest, stats: ProcessStats
) -> Path:
    timestamp = (
        manifest.generated_at.replace(":", "").replace("-", "").replace("+", "Z")
    )
    path = output_root / "manifests" / f"processing-{timestamp}.json"
    write_json(
        path,
        {
            **manifest.model_dump(mode="json"),
            "routing": {
                "mongo_exact": stats.routed_to_mongo,
                "pinecone_semantic": stats.routed_to_pinecone,
                "both_fallback": stats.both_fallback,
            },
        },
    )
    return path


def _source_commit(records: list[NormalizedRecord]) -> str:
    if not records:
        return "unknown"
    return records[0].source_trace.source_commit
