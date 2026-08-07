"""Canonical Mamba-JEPA configuration and training facade.

The training loop, checkpoint schema, metrics, and all-position objective are
shared with Transformer-JEPA.  This module contributes only the Mamba config
surface plus the finite-parameter checks from the source trainer.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

import othello_thesis.training.jepa as _base
from othello_thesis.models.jepa_backbone import normalize_encoder_architecture
from othello_thesis.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_thesis.training._jepa_runner import _run_training as _run_shared_training


_BASE_BUILD_PARSER = _base.build_parser
_BASE_NAMESPACE_TO_TRAIN_CONFIG = _base.namespace_to_train_config
_BASE_VALIDATE_TRAIN_CONFIG = _base.validate_train_config
_BASE_DISPATCH_TRAIN_ONE_CHUNK = _base.dispatch_train_one_chunk
_BASE_LOAD_MODEL_STATE = _base.load_model_state


@dataclass
class MambaJEPATrainConfig(_base.TrainConfig):
    """Canonical JEPA training config extended with Mamba backbone fields."""

    encoder_architecture: str = "mamba"
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mamba_backend: str = "mamba_ssm"


_MAMBA_KEYS = {
    "encoder_architecture",
    "d_state",
    "d_conv",
    "expand",
    "mamba_backend",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the Mamba parser by extending the shared canonical parser."""
    parser = _BASE_BUILD_PARSER()
    parser.prog = "train_mamba_jepa.py"
    parser.description = "Config-driven Mamba-JEPA training loop for Othello."
    parser.add_argument(
        "--encoder_architecture",
        type=str,
        default="mamba",
        choices=["mamba", "mamba_ar", "ssm"],
        help="Mamba-JEPA always uses a Mamba-family encoder.",
    )
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument(
        "--mamba_backend",
        type=str,
        default="mamba_ssm",
        choices=["mamba_ssm", "official", "auto", "torch", "fallback", "mamba_py"],
    )
    return parser


def parse_configured_args(
    argv: list[str],
) -> tuple[argparse.Namespace, Path | None, dict[str, Any]]:
    """Resolve YAML plus CLI overrides into one Mamba namespace."""
    launcher = argparse.ArgumentParser(add_help=False)
    launcher.add_argument("--config", type=str, default=None)
    launcher.add_argument(
        "--print_resolved_config",
        "--print-resolved-config",
        action="store_true",
    )
    launcher_args, remaining = launcher.parse_known_args(argv)

    parser = build_parser()
    config_path: Path | None = None
    config_data: dict[str, Any] = {}
    config_cli_args: list[str] = []
    if launcher_args.config is not None:
        config_path = Path(launcher_args.config).expanduser()
        config_data = _base.load_yaml_config(config_path)
        config_cli_args = _base.config_to_cli_args(config_data, parser)

    args = parser.parse_args([*config_cli_args, *remaining])
    args.config = str(config_path) if config_path is not None else None
    args.print_resolved_config = launcher_args.print_resolved_config
    return args, config_path, config_data


def namespace_to_train_config(
    args: argparse.Namespace,
) -> tuple[MambaJEPATrainConfig, bool]:
    """Convert parsed arguments into the normalized Mamba config."""
    arg_dict = vars(args).copy()
    mamba_values = {
        key: arg_dict.pop(key)
        for key in sorted(_MAMBA_KEYS)
        if key in arg_dict
    }
    base_cfg, print_resolved_config = _BASE_NAMESPACE_TO_TRAIN_CONFIG(
        argparse.Namespace(**arg_dict)
    )
    values = asdict(base_cfg)
    values.update(mamba_values)
    values["encoder_architecture"] = normalize_encoder_architecture(
        values.get("encoder_architecture", "mamba")
    )
    if values["encoder_architecture"] != "mamba":
        raise ValueError("train_mamba_jepa.py only supports Mamba encoders.")
    return MambaJEPATrainConfig(**values), print_resolved_config


def build_objective_config(cfg: MambaJEPATrainConfig) -> JEPAConfig:
    """Build the canonical objective config with Mamba fields preserved."""
    if _base.objective_class(cfg) != "jepa":
        raise ValueError(
            "Master_Thesis_Code_Final Mamba-JEPA supports only "
            "objective_class='jepa'."
        )
    values = asdict(cfg)
    values["variant"] = _base.normalize_jepa_variant(cfg.variant)
    values["loss_type"] = _base.normalize_jepa_loss_type(
        cfg.loss_type,
        cfg.variant,
    )
    return JEPAConfig(
        **_base.filter_kwargs_for_dataclass(JEPAConfig, values)
    )


def build_objective_model(
    cfg: MambaJEPATrainConfig,
) -> tuple[torch.nn.Module, JEPAConfig]:
    """Preserve the legacy programmatic construction helper."""
    objective_cfg = build_objective_config(cfg)
    return OthelloJEPA(objective_cfg), objective_cfg


def validate_train_config(cfg: MambaJEPATrainConfig) -> None:
    """Validate shared invariants plus the Mamba encoder constraints."""
    if normalize_encoder_architecture(cfg.encoder_architecture) != "mamba":
        raise ValueError("Mamba-JEPA configs must set encoder_architecture: mamba")
    if cfg.precision == "fp16":
        raise ValueError(
            "Mamba-JEPA runs must use bf16 or fp32. The v1 fp16 run produced "
            "non-finite Mamba weights, so train_mamba_jepa.py rejects fp16 "
            "to avoid silently saving corrupted checkpoints."
        )
    _BASE_VALIDATE_TRAIN_CONFIG(cfg)


def _assert_model_finite(model: torch.nn.Module, *, where: str) -> None:
    raw_model = _base.unwrap_model(model)
    bad: list[str] = []
    with torch.no_grad():
        for name, parameter in raw_model.named_parameters():
            if parameter.is_floating_point() and not torch.isfinite(parameter).all():
                bad.append(name)
                if len(bad) >= 8:
                    break
    if bad:
        raise FloatingPointError(
            f"Mamba-JEPA model contains non-finite parameters after {where}: "
            + ", ".join(bad)
        )


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    _BASE_LOAD_MODEL_STATE(model, state)
    _assert_model_finite(model, where="checkpoint load")


def dispatch_train_one_chunk(*args, **kwargs):
    model = kwargs.get("model") if "model" in kwargs else args[0]
    cfg = kwargs.get("cfg") if "cfg" in kwargs else args[4]
    profile = bool(cfg.max_batches_per_chunk)
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    finite_started_at = time.perf_counter()
    _assert_model_finite(model, where="training chunk start")
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    finite_start_seconds = time.perf_counter() - finite_started_at
    result = _BASE_DISPATCH_TRAIN_ONE_CHUNK(*args, **kwargs)
    if profile and torch.cuda.is_available():
        torch.cuda.synchronize()
    finite_started_at = time.perf_counter()
    _assert_model_finite(model, where="training chunk")
    if profile:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            "  mamba finite-check profile: "
            f"chunk_start={finite_start_seconds:.3f}s "
            f"chunk_end={time.perf_counter() - finite_started_at:.3f}s",
            flush=True,
        )
    return result


def _run_training(
    cfg: MambaJEPATrainConfig,
    model_factory: _base.ModelFactory | None = None,
    *,
    _source_config: Path | None = None,
    _source_data: dict[str, Any] | None = None,
    _validated: bool = False,
) -> None:
    """Train one resolved Mamba-JEPA config with one post-seed factory call."""
    if not _validated:
        validate_train_config(cfg)
    _run_shared_training(
        cfg,
        model_factory=model_factory,
        _source_config=_source_config,
        _source_data=_source_data,
        _objective_config_factory=build_objective_config,
        _train_dispatch=dispatch_train_one_chunk,
        _state_loader=load_model_state,
        _validated=True,
    )


def run_training(
    cfg: MambaJEPATrainConfig,
    model_factory: _base.ModelFactory | None = None,
) -> None:
    """Train one resolved Mamba config through the notebook-facing API."""
    _run_training(cfg, model_factory=model_factory)


def main(argv: list[str] | None = None) -> None:
    """Mamba CLI with the exact YAML/override resolution contract."""
    cli_argv = sys.argv[1:] if argv is None else argv
    args, config_path, config_data = parse_configured_args(cli_argv)
    cfg, print_resolved_config = namespace_to_train_config(args)
    validate_train_config(cfg)
    if print_resolved_config:
        print(yaml.safe_dump({"resolved_args": asdict(cfg)}, sort_keys=False))
        return
    _run_training(
        cfg,
        _source_config=config_path,
        _source_data=config_data,
        _validated=True,
    )


# Shared helpers retained at their historical public import surface.
build_model_from_checkpoint = _base.build_model_from_checkpoint
config_to_cli_args = _base.config_to_cli_args
filter_kwargs_for_dataclass = _base.filter_kwargs_for_dataclass
forward_all_position_v1 = _base.forward_all_position_v1
load_yaml_config = _base.load_yaml_config
objective_class = _base.objective_class
objective_metric_keys = _base.objective_metric_keys


if __name__ == "__main__":
    main()
