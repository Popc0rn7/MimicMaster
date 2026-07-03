"""PHB normalization pipeline."""

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


def build_phb_records(
    source: FiveEToolsSource, stats: RenderStats, limit: int | None = None
) -> Iterator[NormalizedRecord]:
    yield from iter_book_records(
        book_prefix="phb",
        source_book="PHB",
        record_type="rule_section",
        sections=source.iter_book_sections("book/book-phb.json"),
        stats=stats,
        limit=limit,
    )

    yield from _collection(
        source, stats, "spells/spells-phb.json", "spell", "spell", "PHB", limit
    )

    for relative_path in source.class_files():
        yield from _collection(
            source, stats, relative_path, "class", "class", "PHB", limit
        )
        yield from _collection(
            source, stats, relative_path, "subclass", "subclass", "PHB", limit
        )
        yield from _collection(
            source,
            stats,
            relative_path,
            "classFeature",
            "class_feature",
            "PHB",
            limit,
        )
        yield from _collection(
            source,
            stats,
            relative_path,
            "subclassFeature",
            "subclass_feature",
            "PHB",
            limit,
        )

    for path, key, record_type in [
        ("races.json", "race", "race"),
        ("races.json", "subrace", "subrace"),
        ("backgrounds.json", "background", "background"),
        ("feats.json", "feat", "feat"),
        ("items-base.json", "baseitem", "equipment"),
        ("items.json", "item", "item"),
        ("items.json", "itemGroup", "item_group"),
        ("conditionsdiseases.json", "condition", "condition"),
        ("conditionsdiseases.json", "disease", "disease"),
        ("conditionsdiseases.json", "status", "status"),
        ("optionalfeatures.json", "optionalfeature", "optional_feature"),
        ("variantrules.json", "variantrule", "variant_rule"),
    ]:
        yield from _collection(source, stats, path, key, record_type, "PHB", limit)


def _collection(
    source: FiveEToolsSource,
    stats: RenderStats,
    path: str,
    key: str,
    record_type: str,
    source_book: str,
    limit: int | None,
) -> Iterator[NormalizedRecord]:
    for _, record, trace in limited(source.iter_collection_records(path, key), limit):
        if not is_source(record, source_book):
            continue
        normalized = make_entity_record(
            book_prefix=source_book.lower(),
            source_book=source_book,
            record_type=record_type,
            record=record,
            trace=trace,
            stats=stats,
            section_path=[record_type.replace("_", " ").title()],
        )
        if normalized is not None:
            yield normalized
