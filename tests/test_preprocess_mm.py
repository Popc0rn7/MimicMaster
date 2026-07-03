"""Tests for Monster Manual normalization pipeline."""

from pathlib import Path

from utils.preprocess.pipelines.mm import build_mm_records
from utils.preprocess.renderers import RenderStats
from utils.preprocess.sources.fiveetools import FiveEToolsSource


def test_mm_pipeline_keeps_monsters_structured_and_fluff_separate() -> None:
    source = FiveEToolsSource(Path("submodules/5etools-cn/data"))
    stats = RenderStats()

    records = list(build_mm_records(source, stats=stats, limit=40))

    monster = next(r for r in records if r.record_type == "monster")
    assert monster.content.kind == "structured_entity"
    assert {"ac", "hp", "speed", "cr", "actions"}.issubset(monster.content.structured)
    assert monster.content.structured["actions"]
    assert any(r.record_type == "monster_fluff" for r in records)
    assert any(r.record_type == "monster_section" for r in records)
    assert {r.source_book for r in records} == {"MM"}
