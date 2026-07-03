"""Tests for DMG normalization pipeline."""

from pathlib import Path

from utils.preprocess.pipelines.dmg import build_dmg_records
from utils.preprocess.renderers import RenderStats
from utils.preprocess.sources.fiveetools import FiveEToolsSource


def test_dmg_pipeline_generates_sections_and_independent_tables() -> None:
    source = FiveEToolsSource(Path("submodules/5etools-cn/data"))
    stats = RenderStats()

    records = list(build_dmg_records(source, stats=stats, limit=80))

    assert any(r.record_type == "dm_section" for r in records)
    table = next(r for r in records if r.record_type == "table")
    assert table.content.kind == "table"
    assert table.content.tables
    assert table.source_book == "DMG"
    assert "XDMG" not in {r.source_book for r in records}
