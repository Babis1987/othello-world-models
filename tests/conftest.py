"""Test paths shared by final and cross-repository equivalence tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
FINAL_SRC = FINAL_ROOT / "src"
LEGACY_ROOT = Path(
    os.environ.get(
        "OTHELLO_LEGACY_REPO",
        FINAL_ROOT.parent / "Master_Thesis_Code",
    )
).resolve()


def add_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


add_import_path(FINAL_SRC)
if LEGACY_ROOT.is_dir():
    add_import_path(LEGACY_ROOT)

