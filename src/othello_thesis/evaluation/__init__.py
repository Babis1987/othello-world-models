"""Evaluation entry points used by the thesis notebooks."""

from othello_thesis.evaluation.unified import (
    PROTOCOL_ID,
    EvaluationSplit,
    FrozenEncoderNextMoveHead,
    UnifiedEvalConfig,
    UnifiedEvaluator,
    build_evaluation_split,
)
from othello_thesis.evaluation.thesis import (
    SUITE_ID,
    EvaluationCase,
    ModelNotReadyError,
    prepare_evaluation,
    resolve_case,
    run_complete_evaluation,
)

__all__ = [
    "PROTOCOL_ID",
    "EvaluationSplit",
    "FrozenEncoderNextMoveHead",
    "UnifiedEvalConfig",
    "UnifiedEvaluator",
    "build_evaluation_split",
    "SUITE_ID",
    "EvaluationCase",
    "ModelNotReadyError",
    "prepare_evaluation",
    "resolve_case",
    "run_complete_evaluation",
]
