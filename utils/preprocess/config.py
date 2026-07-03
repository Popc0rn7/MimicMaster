"""Configuration constants for preprocessing."""

from __future__ import annotations

from pathlib import Path

DEFAULT_INPUT_ROOT = Path("submodules/5etools-cn/data")
DEFAULT_OUTPUT_ROOT = Path("knowledge")
PROVIDER = "5etools-cn"
SOURCE_LICENSE = "CC BY-NC-SA 4.0"
TARGET_SOURCES = {"phb": "PHB", "dmg": "DMG", "mm": "MM"}
