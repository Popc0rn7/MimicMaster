"""DMG normalization pipeline."""

from __future__ import annotations

from collections.abc import Iterator

from utils.preprocess.models import NormalizedRecord
from utils.preprocess.pipelines.common import (
    is_source,
    iter_book_records,
    limited,
    make_entity_record,
)
from utils.preprocess.renderers import RenderStats
from utils.preprocess.sources.fiveetools import FiveEToolsSource


def build_dmg_records(
    source: FiveEToolsSource, stats: RenderStats, limit: int | None = None
) -> Iterator[NormalizedRecord]:
    yield from iter_book_records(
        book_prefix="dmg",
        source_book="DMG",
        record_type="dm_section",
        sections=source.iter_book_sections("book/book-dmg.json"),
        stats=stats,
        limit=limit,
    )

    for path, key, record_type in [
        ("items.json", "item", "magic_item"),
        ("items.json", "itemGroup", "item_group"),
        ("items-base.json", "baseitem", "equipment"),
        ("rewards.json", "reward", "reward"),
        ("trapshazards.json", "trap", "trap"),
        ("trapshazards.json", "hazard", "hazard"),
        ("objects.json", "object", "object"),
        ("conditionsdiseases.json", "disease", "disease"),
        ("variantrules.json", "variantrule", "variant_rule"),
    ]:
        yield from _collection(source, stats, path, key, record_type, limit)


def _collection(
    source: FiveEToolsSource,
    stats: RenderStats,
    path: str,
    key: str,
    record_type: str,
    limit: int | None,
) -> Iterator[NormalizedRecord]:
    for _, record, trace in limited(source.iter_collection_records(path, key), limit):
        if not is_source(record, "DMG"):
            continue
        normalized = make_entity_record(
            book_prefix="dmg",
            source_book="DMG",
            record_type=record_type,
            record=record,
            trace=trace,
            stats=stats,
            section_path=[record_type.replace("_", " ").title()],
        )
        if normalized is not None:
            yield normalized
