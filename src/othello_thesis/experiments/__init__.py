"""Resolved experiment contracts used by the final notebooks."""

from .canonical import (
    BOARD_SIZES,
    RunLayout,
    build_ar_train_config,
    build_jepa_train_config,
    canonical_config_dir,
    resolve_run_layout,
    validate_canonical_corpus,
)

__all__ = [
    "BOARD_SIZES",
    "RunLayout",
    "build_ar_train_config",
    "build_jepa_train_config",
    "canonical_config_dir",
    "resolve_run_layout",
    "validate_canonical_corpus",
]
