"""Run the no-write Phase 7 smoke for the official thesis Mamba backend."""

from __future__ import annotations

import importlib.metadata as metadata
import random
import sys
import warnings
from pathlib import Path


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from othello_thesis.models.mamba import MambaARConfig, OthelloMambaAR
from othello_thesis.training.ar_runtime import autocast_context


def _validate_environment() -> torch.device:
    """Require the source-recorded production package and CUDA versions."""
    assert str(torch.__version__) == "2.11.0+cu128", torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert metadata.version("mamba-ssm") == "2.3.2.post1"
    assert metadata.version("causal-conv1d") == "1.6.2.post1"
    assert torch.cuda.is_available(), "A CUDA runtime is required"
    assert torch.cuda.is_bf16_supported(), "The CUDA device must support BF16"
    return torch.device("cuda")


def _production_config() -> MambaARConfig:
    """Return the exact board-12 production Mamba geometry."""
    config = MambaARConfig(
        board_size=12,
        n_layers=15,
        d_model=512,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        mlp_hidden_mult=0,
        mamba_backend="mamba_ssm",
    )
    assert (config.vocab_size, config.block_size) == (141, 139)
    return config


def _validate_resume_stable_length_one_path(
    model: OthelloMambaAR,
    device: torch.device,
) -> None:
    """Exercise the official JEPA length-one path, including forced drift."""
    model.eval()
    model.zero_grad(set_to_none=True)
    token = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
    with torch.no_grad(), autocast_context(device, "bf16"):
        logits, loss = model(token)
    assert loss is None
    assert logits.shape == (1, 1, model.config.vocab_size)
    assert logits.dtype == torch.bfloat16
    assert all(block._length_one_fast_path_checked for block in model.blocks)
    assert all(block._length_one_fast_path_enabled for block in model.blocks)

    # Reproduce the resume failure mode deterministically.  A diagnostic
    # fused-kernel mismatch must warn, but it must never switch this block to
    # the general selective-scan path.
    block = model.blocks[0]
    block._length_one_fast_path_checked = False
    original_forward = block.mixer.forward

    def drifted_reference(hidden_states: torch.Tensor) -> torch.Tensor:
        return original_forward(hidden_states) + 0.25

    block.mixer.forward = drifted_reference
    hidden = torch.randn(
        2,
        1,
        model.config.d_model,
        device=device,
        requires_grad=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with autocast_context(device, "bf16"):
            output = block(hidden)
            fast_path_loss = output.float().square().mean()
    block.mixer.forward = original_forward

    assert any(
        "resumed runs use the same computation" in str(item.message)
        for item in caught
    ), [str(item.message) for item in caught]
    assert block._length_one_fast_path_checked is True
    assert block._length_one_fast_path_enabled is True
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all().item()

    fast_path_loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all().item()
    out_proj_grad = block.mixer.out_proj.weight.grad
    assert out_proj_grad is not None and torch.isfinite(out_proj_grad).all().item()


def main() -> None:
    """Construct and update the production model without writing artifacts."""
    device = _validate_environment()
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = _production_config()
    model = OthelloMambaAR(config).to(device)
    model.train()
    assert model.num_parameters() == 25_652_736

    mixers = [block.mixer for block in model.blocks]
    mixer_classes = sorted(
        {f"{type(mixer).__module__}.{type(mixer).__name__}" for mixer in mixers}
    )
    assert len(mixers) == 15
    assert all(
        type(mixer).__name__ == "Mamba"
        and type(mixer).__module__.startswith("mamba_ssm.")
        for mixer in mixers
    ), mixer_classes
    assert mixer_classes == ["mamba_ssm.modules.mamba_simple.Mamba"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, config.block_size + 1),
        device=device,
    )
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    updated_parameter = model.lm_head.weight
    before = updated_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, "bf16"):
        logits, loss = model(inputs, targets)

    assert logits.shape == (1, 139, 141)
    assert logits.dtype == torch.bfloat16
    assert loss is not None and torch.isfinite(loss).item()
    loss.backward()
    assert updated_parameter.grad is not None
    assert torch.isfinite(updated_parameter.grad).all().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    assert torch.isfinite(grad_norm).item()

    optimizer.step()
    torch.cuda.synchronize()
    max_update = (updated_parameter.detach() - before).abs().max()
    assert torch.isfinite(max_update).item()
    assert max_update.item() > 0.0

    _validate_resume_stable_length_one_path(model, device)
    torch.cuda.synchronize()

    print("OFFICIAL_MAMBA_SMOKE_PASS")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)}"
    )
    print(
        f"mamba_ssm={metadata.version('mamba-ssm')} "
        f"causal_conv1d={metadata.version('causal-conv1d')}"
    )
    print(f"mixer={mixer_classes[0]} layers={len(mixers)}")
    print(
        f"params={model.num_parameters()} "
        f"vocab={config.vocab_size} block={config.block_size}"
    )
    print(f"logits={tuple(logits.shape)} dtype={logits.dtype}")
    print(
        f"length_one_layers={len(model.blocks)} "
        "forced_parity_drift=PASS backward=PASS"
    )
    print(
        f"loss={loss.detach().float().item():.8f} "
        f"grad_norm={grad_norm.detach().float().item():.8f} "
        f"max_lm_head_update={max_update.detach().float().item():.8e}"
    )


if __name__ == "__main__":
    main()
