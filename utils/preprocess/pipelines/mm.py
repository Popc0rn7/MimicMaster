"""Monster Manual normalization pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from utils.preprocess.models import NormalizedContent, NormalizedRecord
from utils.preprocess.pipelines.common import (
    is_source,
    iter_book_records,
    limited,
    make_entity_record,
    slugify,
)
from utils.preprocess.renderers import RenderStats, extract_tables, render_entries
from utils.preprocess.sources.fiveetools import FiveEToolsSource


def build_mm_records(
    source: FiveEToolsSource, stats: RenderStats, limit: int | None = None
) -> Iterator[NormalizedRecord]:
    yield from _monsters(source, stats, limit)
    yield from _fluff(source, stats, limit)
    yield from iter_book_records(
        book_prefix="mm",
        source_book="MM",
        record_type="monster_section",
        sections=source.iter_book_sections("book/book-mm.json"),
        stats=stats,
        limit=limit,
    )


def _monsters(
    source: FiveEToolsSource, stats: RenderStats, limit: int | None
) -> Iterator[NormalizedRecord]:
    for _, record, trace in limited(
        source.iter_collection_records("bestiary/bestiary-mm.json", "monster"), limit
    ):
        if not is_source(record, "MM"):
            continue
        entries = []
        for key in ("trait", "action", "reaction", "legendary", "variant"):
            entries.extend(record.get(key, []))
        rendered = render_entries(entries, stats)
        if not rendered:
            rendered = render_entries(
                [
                    record.get("name"),
                    f"AC: {record.get('ac')}",
                    f"HP: {record.get('hp')}",
                    f"CR: {record.get('cr')}",
                ],
                stats,
            )
        if not rendered:
            stats.empty_records += 1
            continue
        title = str(record.get("name") or record.get("ENG_name") or "Monster")
        eng_title = str(record.get("ENG_name")) if record.get("ENG_name") else None
        yield NormalizedRecord(
            id=f"mm:monster:{slugify(eng_title, title)}",
            source_book="MM",
            record_type="monster",
            title=title,
            eng_title=eng_title,
            page=record.get("page"),
            section_path=["Monsters"],
            source_trace=trace,
            content=NormalizedContent(
                kind="structured_entity",
                entries=entries,
                structured=_monster_structured(record),
                tables=extract_tables(entries, stats),
                rendered_text=rendered,
            ),
        )


def _fluff(
    source: FiveEToolsSource, stats: RenderStats, limit: int | None
) -> Iterator[NormalizedRecord]:
    for _, record, trace in limited(
        source.iter_collection_records(
            "bestiary/fluff-bestiary-mm.json", "monsterFluff"
        ),
        limit,
    ):
        if not is_source(record, "MM"):
            continue
        normalized = make_entity_record(
            book_prefix="mm",
            source_book="MM",
            record_type="monster_fluff",
            record=record,
            trace=trace,
            stats=stats,
            section_path=["Monster Fluff"],
        )
        if normalized is not None:
            yield normalized


def _monster_structured(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "ac": record.get("ac"),
        "hp": record.get("hp"),
        "speed": record.get("speed"),
        "abilities": {
            "str": record.get("str"),
            "dex": record.get("dex"),
            "con": record.get("con"),
            "int": record.get("int"),
            "wis": record.get("wis"),
            "cha": record.get("cha"),
        },
        "saves": record.get("save", {}),
        "skills": record.get("skill", {}),
        "resistances": record.get("resist", []),
        "immunities": {
            "damage": record.get("immune", []),
            "condition": record.get("conditionImmune", []),
        },
        "senses": record.get("senses", []),
        "passive": record.get("passive"),
        "languages": record.get("languages", []),
        "cr": record.get("cr"),
        "traits": record.get("trait", []),
        "actions": record.get("action", []),
        "reactions": record.get("reaction", []),
        "legendary": record.get("legendary", []),
        "spellcasting": record.get("spellcasting", []),
        "variants": record.get("variant", []),
    }
