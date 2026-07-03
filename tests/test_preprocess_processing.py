"""Tests for processing normalized records into retrieval-specific outputs."""

import json
from pathlib import Path

from utils.preprocess.models import NormalizedContent, NormalizedRecord, SourceTrace
from utils.preprocess.processing import (
    ProcessStats,
    process_records,
    process_to_outputs,
)


def test_process_records_routes_structured_entities_to_mongo() -> None:
    records = [
        _record(
            record_id="phb:spell:fireball",
            source_book="PHB",
            record_type="spell",
            title="火球术",
            eng_title="Fireball",
            kind="structured_entity",
            structured={"level": 3, "school": "V", "damageInflict": ["fire"]},
            text="一颗明亮的闪光从你的指尖飞向指定地点。",
        )
    ]

    processed = process_records(records, stats=ProcessStats())

    assert len(processed.mongo_entities) == 1
    assert processed.pinecone_chunks == []
    entity = processed.mongo_entities[0]
    assert entity.id == "phb:spell:fireball"
    assert entity.retrieval_mode == "mongo_exact"
    assert entity.aliases == ["火球术", "Fireball"]
    assert "spell" in entity.record_tags
    assert "damage:fire" in entity.mechanic_tags


def test_process_records_routes_narrative_to_self_contained_pinecone_chunk() -> None:
    records = [
        _record(
            record_id="phb:rule_section:cover:0",
            source_book="PHB",
            record_type="rule_section",
            title="掩护",
            eng_title=None,
            kind="narrative",
            structured={},
            text="墙壁、树木、生物和其他障碍物都能在战斗中提供掩护。",
            section_path=["战斗", "掩护"],
            page=196,
        )
    ]

    processed = process_records(records, stats=ProcessStats())

    assert processed.mongo_entities == []
    assert len(processed.pinecone_chunks) == 1
    chunk = processed.pinecone_chunks[0]
    assert chunk.id == "pinecone:phb:rule_section:cover:0:0"
    assert chunk.retrieval_mode == "pinecone_semantic"
    assert chunk.text == "墙壁、树木、生物和其他障碍物都能在战斗中提供掩护。"
    assert chunk.source_book == "PHB"
    assert chunk.page == 196
    assert "combat" in chunk.semantic_tags


def test_process_records_routes_tables_to_mongo_for_exact_lookup() -> None:
    records = [
        _record(
            record_id="dmg:table:treasure:0",
            source_book="DMG",
            record_type="table",
            title="宝藏表",
            eng_title=None,
            kind="table",
            structured={},
            text="宝藏表\n骰值 | 结果\n1 | 金币",
            tables=[
                {
                    "caption": "宝藏表",
                    "col_labels": ["骰值", "结果"],
                    "rows": [["1", "金币"]],
                }
            ],
        )
    ]

    processed = process_records(records, stats=ProcessStats())

    assert len(processed.mongo_entities) == 1
    assert processed.pinecone_chunks == []
    entity = processed.mongo_entities[0]
    assert entity.retrieval_mode == "mongo_exact"
    assert entity.tables[0]["rows"] == [["1", "金币"]]
    assert "table" in entity.record_tags


def test_process_to_outputs_writes_mongo_and_pinecone_jsonl(tmp_path: Path) -> None:
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    _write_jsonl(
        normalized_dir / "phb.jsonl",
        [
            _record(
                record_id="phb:spell:fireball",
                source_book="PHB",
                record_type="spell",
                title="火球术",
                eng_title="Fireball",
                kind="structured_entity",
                structured={"level": 3},
                text="火球术描述。",
            ),
            _record(
                record_id="phb:rule_section:cover:0",
                source_book="PHB",
                record_type="rule_section",
                title="掩护",
                eng_title=None,
                kind="narrative",
                structured={},
                text="掩护规则描述。",
            ),
        ],
    )

    manifest = process_to_outputs(normalized_dir=normalized_dir, output_root=tmp_path)

    mongo_lines = (tmp_path / "mongo" / "entities.jsonl").read_text(encoding="utf-8")
    pinecone_lines = (tmp_path / "pinecone" / "chunks.jsonl").read_text(
        encoding="utf-8"
    )
    assert len(mongo_lines.strip().splitlines()) == 1
    assert len(pinecone_lines.strip().splitlines()) == 1
    assert manifest.outputs["mongo_entities"]["record_count"] == 1
    assert manifest.outputs["pinecone_chunks"]["record_count"] == 1


def test_process_to_outputs_does_not_overwrite_normalize_latest_manifest(
    tmp_path: Path,
) -> None:
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    normalize_latest = manifests_dir / "latest.json"
    normalize_latest.write_text('{"sources":["phb","dmg","mm"]}\n', encoding="utf-8")
    _write_jsonl(
        normalized_dir / "phb.jsonl",
        [
            _record(
                record_id="phb:rule_section:cover:0",
                source_book="PHB",
                record_type="rule_section",
                title="掩护",
                eng_title=None,
                kind="narrative",
                structured={},
                text="掩护规则描述。",
            )
        ],
    )

    process_to_outputs(normalized_dir=normalized_dir, output_root=tmp_path)

    assert json.loads(normalize_latest.read_text(encoding="utf-8")) == {
        "sources": ["phb", "dmg", "mm"]
    }
    assert (manifests_dir / "processing-latest.json").exists()


def _record(
    *,
    record_id: str,
    source_book: str,
    record_type: str,
    title: str,
    eng_title: str | None,
    kind: str,
    structured: dict,
    text: str,
    section_path: list[str] | None = None,
    page: int | None = 1,
    tables: list[dict] | None = None,
) -> NormalizedRecord:
    return NormalizedRecord(
        id=record_id,
        source_book=source_book,
        record_type=record_type,
        title=title,
        eng_title=eng_title,
        page=page,
        section_path=section_path or [title],
        source_trace=SourceTrace(
            provider="5etools-cn",
            source_file="data/example.json",
            source_key="data",
            source_index=0,
            source_commit="abc",
            license="CC BY-NC-SA 4.0",
        ),
        content=NormalizedContent(
            kind=kind,
            entries=[],
            structured=structured,
            tables=tables or [],
            rendered_text=text,
        ),
    )


def _write_jsonl(path: Path, records: list[NormalizedRecord]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            file.write("\n")
