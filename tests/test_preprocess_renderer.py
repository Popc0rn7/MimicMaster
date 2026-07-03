"""Tests for 5etools entry rendering."""

from utils.preprocess.renderers import RenderStats, render_entries, render_inline


def test_render_inline_strips_common_5etools_tags() -> None:
    stats = RenderStats()

    rendered = render_inline(
        "{@b Bold} {@i Italic} {@spell 火球术|PHB} {@damage 8d6} {@dc 15} {@skill 察觉}",
        stats,
    )

    assert rendered == "Bold Italic 火球术 8d6 DC 15 察觉"
    assert stats.unknown_tags == {}


def test_render_inline_preserves_unknown_tag_display_text_and_counts_it() -> None:
    stats = RenderStats()

    rendered = render_inline("{@deity 阿斯蒙蒂斯|破晓战争|DMG}", stats)

    assert rendered == "阿斯蒙蒂斯"
    assert stats.unknown_tags == {"deity": 1}


def test_render_entries_handles_tables_and_lists() -> None:
    stats = RenderStats()
    text = render_entries(
        [
            "开场文字",
            {"type": "list", "items": ["第一项", "{@b 第二项}"]},
            {
                "type": "table",
                "caption": "示例表",
                "colLabels": ["A", "B"],
                "rows": [["{@i 甲}", "乙"]],
            },
        ],
        stats,
    )

    assert "开场文字" in text
    assert "- 第一项" in text
    assert "第二项" in text
    assert "示例表" in text
    assert "A | B" in text
    assert "甲 | 乙" in text
