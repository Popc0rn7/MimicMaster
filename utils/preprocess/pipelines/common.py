"""Shared helpers for source-book normalization pipelines."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from typing import Any

from utils.preprocess.models import NormalizedContent, NormalizedRecord, SourceTrace
from utils.preprocess.renderers import (
    RenderStats,
    extract_tables,
    normalize_table,
    render_entries,
)

SLUG_RE = re.compile(r"[^a-z0-9]+")
SKIP_BOOK_TYPES = {"image", "internal", "gallery"}


def slugify(*values: str | None) -> str:
    """Build a stable ASCII slug, hashing non-ASCII titles when needed."""

    raw = next((value for value in values if value), "record")
    slug = SLUG_RE.sub("-", raw.lower()).strip("-")
    if slug:
        return slug
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"record-{digest}"


def limited(
    records: Iterable[tuple[int, dict[str, Any], SourceTrace]], limit: int | None
):
    count = 0
    for item in records:
        if limit is not None and count >= limit:
            break
        count += 1
        yield item


def is_source(record: dict[str, Any], source_book: str) -> bool:
    return record.get("source") == source_book


def make_entity_record(
    *,
    book_prefix: str,
    source_book: str,
    record_type: str,
    record: dict[str, Any],
    trace: SourceTrace,
    stats: RenderStats,
    id_parts: list[str] | None = None,
    structured: dict[str, Any] | None = None,
    section_path: list[str] | None = None,
) -> NormalizedRecord | None:
    entries = record.get("entries", [])
    rendered = render_entries(entries, stats)
    if not rendered:
        rendered = render_entries(_structured_fallback(record), stats)
    if not rendered:
        stats.empty_records += 1
        return None
    title = str(record.get("name") or record.get("ENG_name") or "Untitled")
    eng_title = record.get("ENG_name")
    slug_parts = id_parts or [eng_title, title]
    return NormalizedRecord(
        id=f"{book_prefix}:{record_type}:{slugify(*slug_parts)}",
        source_book=source_book,
        record_type=record_type,
        title=title,
        eng_title=str(eng_title) if eng_title else None,
        page=record.get("page"),
        section_path=section_path or [],
        source_trace=trace,
        content=NormalizedContent(
            kind="structured_entity",
            entries=entries if isinstance(entries, list) else [entries],
            structured=(
                structured if structured is not None else _copy_structured(record)
            ),
            tables=extract_tables(entries, stats),
            rendered_text=rendered,
        ),
    )


def make_table_record(
    *,
    book_prefix: str,
    source_book: str,
    record_type: str = "table",
    title: str,
    table: dict[str, Any],
    trace: SourceTrace,
    stats: RenderStats,
    section_path: list[str],
    index: int,
    page: int | None = None,
) -> NormalizedRecord | None:
    normalized = normalize_table(table, stats)
    rendered = render_entries(table, stats)
    if not rendered:
        stats.empty_records += 1
        return None
    return NormalizedRecord(
        id=f"{book_prefix}:table:{slugify(title)}:{index}",
        source_book=source_book,
        record_type=record_type,
        title=title,
        eng_title=None,
        page=page,
        section_path=section_path,
        source_trace=trace,
        content=NormalizedContent(
            kind="table",
            entries=[table],
            structured={},
            tables=[normalized],
            rendered_text=rendered,
        ),
    )


def iter_book_records(
    *,
    book_prefix: str,
    source_book: str,
    record_type: str,
    sections: Iterable[tuple[int, dict[str, Any], SourceTrace]],
    stats: RenderStats,
    limit: int | None = None,
) -> Iterator[NormalizedRecord]:
    emitted = 0
    table_index = 0
    for index, section, trace in sections:
        if limit is not None and index >= limit:
            break
        path = [str(section.get("name") or section.get("ENG_name") or "Untitled")]
        for record in _walk_book_node(
            book_prefix=book_prefix,
            source_book=source_book,
            record_type=record_type,
            node=section,
            trace=trace,
            stats=stats,
            section_path=path,
            record_index=emitted,
            table_index=table_index,
            page=section.get("page"),
        ):
            if record.record_type == "table":
                table_index += 1
            else:
                emitted += 1
            yield record


def _walk_book_node(
    *,
    book_prefix: str,
    source_book: str,
    record_type: str,
    node: Any,
    trace: SourceTrace,
    stats: RenderStats,
    section_path: list[str],
    record_index: int,
    table_index: int,
    page: int | None,
) -> Iterator[NormalizedRecord]:
    if isinstance(node, str):
        rendered = render_entries([node], stats)
        if not rendered:
            stats.empty_records += 1
            return
        title = section_path[-1] if section_path else "Section"
        yield NormalizedRecord(
            id=f"{book_prefix}:{record_type}:{slugify(*section_path)}:{record_index}",
            source_book=source_book,
            record_type=record_type,
            title=title,
            eng_title=None,
            page=page,
            section_path=section_path,
            source_trace=trace,
            content=NormalizedContent(
                kind="narrative",
                entries=[node],
                structured={},
                tables=[],
                rendered_text=rendered,
            ),
        )
        return
    if isinstance(node, list):
        current = record_index
        for item in node:
            for record in _walk_book_node(
                book_prefix=book_prefix,
                source_book=source_book,
                record_type=record_type,
                node=item,
                trace=trace,
                stats=stats,
                section_path=section_path,
                record_index=current,
                table_index=table_index,
                page=page,
            ):
                current += 1
                yield record
        return
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type in SKIP_BOOK_TYPES:
        stats.skipped_items += 1
        return
    if node_type == "table":
        title = str(
            node.get("caption") or (section_path[-1] if section_path else "Table")
        )
        table_record = make_table_record(
            book_prefix=book_prefix,
            source_book=source_book,
            title=title,
            table=node,
            trace=trace,
            stats=stats,
            section_path=section_path,
            index=table_index,
            page=page,
        )
        if table_record is not None:
            yield table_record
        return

    next_path = section_path
    name = node.get("name")
    if name and str(name) not in section_path:
        next_path = [*section_path, str(name)]

    entries = node.get("entries")
    if entries is None:
        return
    rendered = render_entries(entries, stats)
    tables = extract_tables(entries, stats)
    if rendered:
        title = str(node.get("name") or (next_path[-1] if next_path else "Section"))
        yield NormalizedRecord(
            id=f"{book_prefix}:{record_type}:{slugify(*next_path)}:{record_index}",
            source_book=source_book,
            record_type=record_type,
            title=title,
            eng_title=str(node.get("ENG_name")) if node.get("ENG_name") else None,
            page=node.get("page", page),
            section_path=next_path,
            source_trace=trace,
            content=NormalizedContent(
                kind="narrative",
                entries=entries if isinstance(entries, list) else [entries],
                structured={},
                tables=tables,
                rendered_text=rendered,
            ),
        )
    else:
        stats.empty_records += 1

    for entry in entries if isinstance(entries, list) else [entries]:
        if isinstance(entry, dict) and entry.get("type") == "table":
            table_record = make_table_record(
                book_prefix=book_prefix,
                source_book=source_book,
                title=str(entry.get("caption") or title),
                table=entry,
                trace=trace,
                stats=stats,
                section_path=next_path,
                index=table_index,
                page=node.get("page", page),
            )
            if table_record is not None:
                yield table_record
        elif isinstance(entry, dict) and entry.get("entries"):
            yield from _walk_book_node(
                book_prefix=book_prefix,
                source_book=source_book,
                record_type=record_type,
                node=entry,
                trace=trace,
                stats=stats,
                section_path=next_path,
                record_index=record_index + 1,
                table_index=table_index + 1,
                page=node.get("page", page),
            )


def _copy_structured(record: dict[str, Any]) -> dict[str, Any]:
    ignored = {"entries", "fluff", "images", "soundClip"}
    return {key: value for key, value in record.items() if key not in ignored}


def _structured_fallback(record: dict[str, Any]) -> list[str]:
    parts = []
    for key in ("name", "ENG_name", "source", "page", "level", "type", "rarity"):
        if key in record:
            parts.append(f"{key}: {record[key]}")
    return parts
