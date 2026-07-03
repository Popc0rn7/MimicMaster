"""Tests for normalized preprocess models."""

import pytest
from pydantic import ValidationError

from utils.preprocess.models import NormalizedContent, NormalizedRecord, SourceTrace


def test_normalized_record_requires_common_shell() -> None:
    record = NormalizedRecord(
        id="phb:spell:acid-splash",
        source_book="PHB",
        record_type="spell",
        title="酸液飞溅",
        eng_title="Acid Splash",
        page=211,
        section_path=["法术", "戏法"],
        source_trace=SourceTrace(
            provider="5etools-cn",
            source_file="data/spells/spells-phb.json",
            source_key="spell",
            source_index=0,
            source_commit="abc123",
            license="CC BY-NC-SA 4.0",
        ),
        content=NormalizedContent(
            kind="structured_entity",
            entries=["造成{@damage 1d6}点强酸伤害。"],
            structured={"level": 0},
            tables=[],
            rendered_text="造成1d6点强酸伤害。",
        ),
    )

    dumped = record.model_dump()

    assert dumped["id"] == "phb:spell:acid-splash"
    assert dumped["content"]["kind"] == "structured_entity"
    assert dumped["source_trace"]["source_file"] == "data/spells/spells-phb.json"


def test_content_kind_is_limited_to_known_values() -> None:
    with pytest.raises(ValidationError):
        NormalizedContent(kind="image", rendered_text="not allowed")
