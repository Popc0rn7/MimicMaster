"""Tests for PHB normalization pipeline."""

from pathlib import Path

from utils.preprocess.pipelines.phb import build_phb_records
from utils.preprocess.renderers import RenderStats
from utils.preprocess.sources.fiveetools import FiveEToolsSource


def test_phb_pipeline_generates_spell_feature_and_section_records() -> None:
    source = FiveEToolsSource(Path("submodules/5etools-cn/data"))
    stats = RenderStats()

    records = list(build_phb_records(source, stats=stats, limit=80))

    assert any(r.record_type == "rule_section" for r in records)
    acid_splash = next(r for r in records if r.id == "phb:spell:acid-splash")
    assert acid_splash.title == "酸液飞溅"
    assert acid_splash.content.kind == "structured_entity"
    assert acid_splash.content.structured["level"] == 0
    assert "1d6" in acid_splash.content.rendered_text
    assert any(r.record_type == "class_feature" for r in records)
    assert {r.source_book for r in records} == {"PHB"}
