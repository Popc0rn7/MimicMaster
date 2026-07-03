"""Tests for preprocess writer helpers."""

from utils.preprocess.models import NormalizedContent, NormalizedRecord, SourceTrace
from utils.preprocess.writers import ensure_unique_ids


def test_ensure_unique_ids_adds_stable_suffixes_to_duplicates() -> None:
    records = [
        _record("phb:rule:same"),
        _record("phb:rule:same"),
        _record("phb:rule:other"),
    ]

    unique = ensure_unique_ids(records)

    assert [record.id for record in unique] == [
        "phb:rule:same",
        "phb:rule:same:2",
        "phb:rule:other",
    ]


def _record(record_id: str) -> NormalizedRecord:
    return NormalizedRecord(
        id=record_id,
        source_book="PHB",
        record_type="rule",
        title="Rule",
        source_trace=SourceTrace(
            provider="5etools-cn",
            source_file="data/example.json",
            source_key="data",
            source_index=0,
            source_commit="abc",
            license="CC BY-NC-SA 4.0",
        ),
        content=NormalizedContent(kind="narrative", rendered_text="text"),
    )
