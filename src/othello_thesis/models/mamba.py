"""Autoregressive Mamba-style sequence model for Othello moves.

The public API intentionally mirrors ``OthelloGPT``:
``forward(idx, targets=None) -> (logits, loss)`` and the model exposes
``wte``, ``wpe``, ``drop``, ``blocks`` and ``ln_f`` so the existing legal
move evaluator and board-state probes can operate on it.

For thesis Mamba-AR experiments the backend is ``mamba_ssm``. The pure
PyTorch backend remains available only for local unit tests or dependency
debugging; it is not the experiment default.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_thesis.data.chunk_dataset import TARGET_PAD


@dataclass
class MambaARConfig:
    """Config for an Othello autoregressive Mamba model."""

    architecture: str = "mamba_ar"
    board_size: int = 8
    n_layers: int = 15
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    mlp_hidden_mult: int = 0
    mamba_backend: str = "mamba_ssm"
    vocab_size: Optional[int] = None
    block_size: Optional[int] = None

    def __post_init__(self) -> None:
        max_moves = self.board_size**2 - 4
        if self.vocab_size is None:
            self.vocab_size = max_moves + 1
        if self.block_size is None:
            self.block_size = max_moves - 1
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive, got {self.d_state}")
        if self.d_conv <= 0:
            raise ValueError(f"d_conv must be positive, got {self.d_conv}")
        if self.expand <= 0:
            raise ValueError(f"expand must be positive, got {self.expand}")
        self.mamba_backend = str(self.mamba_backend).lower().replace("-", "_")


def _load_official_mamba():
    try:
        from mamba_ssm import Mamba  # type: ignore

        return Mamba
    except Exception as first_error:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            return Mamba
        except Exception as second_error:
            raise ImportError(
                "mamba_ssm is not available. Install the optional dependency "
                "with `pip install \"mamba-ssm[causal-conv1d]\" --no-build-isolation`. "
                "The Mamba-AR experiment backend is strict and does not "
                "fall back automatically."
            ) from second_error or first_error



class PScan(torch.autograd.Function):
    """Parallel scan for h[t] = A[t] * h[t-1] + X[t].

    Adapted from alxndrTL/mamba.py's Blelloch-style PyTorch scan. Inputs and
    outputs use shape (B, L, D, N).
    """

    @staticmethod
    def pscan(A: torch.Tensor, X: torch.Tensor) -> None:
        batch, dim, length, _ = A.size()
        num_steps = int(math.log2(length))

        Aa = A
        Xa = X
        for _ in range(num_steps):
            even_length = 2 * (Xa.size(2) // 2)
            Aa = Aa[:, :, :even_length].view(batch, dim, even_length // 2, 2, -1)
            Xa = Xa[:, :, :even_length].view(batch, dim, even_length // 2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1:length:2**k]
            Xa = X[:, :, 2**k - 1:length:2**k]
            even_length = 2 * (Xa.size(2) // 2)

            if even_length < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :even_length].view(batch, dim, even_length // 2, 2, -1)
            Xa = Xa[:, :, :even_length].view(batch, dim, even_length // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        A = A_in.clone().transpose(2, 1)
        X = X_in.clone().transpose(2, 1)
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in: torch.Tensor):
        A_in, X = ctx.saved_tensors
        A = A_in.clone().transpose(2, 1)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output = grad_output_in.transpose(2, 1).flip(2)
        PScan.pscan(A, grad_output)
        grad_output = grad_output.flip(2)
        grad_a = torch.zeros_like(X)
        grad_a[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
        return grad_a.transpose(2, 1), grad_output.transpose(2, 1)


pscan = PScan.apply


class TorchMambaMixer(nn.Module):
    """Pure PyTorch Mamba mixer.

    This follows the reference mamba.py structure: input projection, causal
    depthwise convolution, input-dependent SSM parameters, selective scan,
    multiplicative gate, and output projection. It is slower than mamba-ssm
    but avoids the external CUDA build dependency.
    """

    def __init__(self, config: MambaARConfig):
        super().__init__()
        self.d_model = int(config.d_model)
        self.d_state = int(config.d_state)
        self.d_conv = int(config.d_conv)
        self.expand = int(config.expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_inner,
            1,
        )
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        y = self._ssm(x_branch)
        y = y * F.silu(z_branch)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        delta_b_c = self.x_proj(x)
        delta, b, c = torch.split(
            delta_b_c,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))
        delta_a = torch.exp(delta.unsqueeze(-1) * a)
        delta_b = delta.unsqueeze(-1) * b.unsqueeze(2)
        bx = delta_b * x.unsqueeze(-1)
        hs = pscan(delta_a, bx)
        y = (hs @ c.unsqueeze(-1)).squeeze(3)
        return y + d * x


def build_mamba_mixer(config: MambaARConfig) -> nn.Module:
    backend = config.mamba_backend
    if backend in {"auto", "mamba_ssm", "official"}:
        mamba_cls = _load_official_mamba()
        return mamba_cls(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    if backend in {"fallback", "torch", "mamba_py"}:
        warnings.warn(
            "Using the pure PyTorch Mamba backend. This path is intended "
            "for local unit tests/debugging, not thesis Mamba-AR runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return TorchMambaMixer(config)

    raise ValueError(
        f"Unknown mamba_backend={config.mamba_backend!r}; expected "
        "'mamba_ssm', 'official', 'auto', or 'torch'."
    )


class MambaMLP(nn.Module):
    def __init__(self, config: MambaARConfig):
        super().__init__()
        hidden = int(config.mlp_hidden_mult * config.d_model)
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mamba_length_one_forward(mixer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Exact zero-state Mamba-1 update for an independent length-one sequence.

    The hard-disjoint JEPA target encoder receives one token with a fresh zero
    recurrent state.  For that single timestep the ``A`` recurrence multiplies
    zero and drops out algebraically.  Avoiding the generic selective-scan
    kernel prevents allocation of a ``(B, d_inner, d_state)`` state for tens of
    thousands of independent targets while retaining gradients through every
    parameter that affects the one-step output.
    """
    if hidden_states.ndim != 3 or hidden_states.size(1) != 1:
        raise ValueError(
            "mamba_length_one_forward requires shape (B, 1, D), got "
            f"{tuple(hidden_states.shape)}"
        )
    required = (
        "in_proj", "conv1d", "x_proj", "dt_proj", "D", "out_proj",
        "dt_rank", "d_state",
    )
    missing = [name for name in required if not hasattr(mixer, name)]
    if missing:
        raise TypeError("Unsupported Mamba-1 mixer; missing: " + ", ".join(missing))

    dtype = hidden_states.dtype
    x_branch, z_branch = mixer.in_proj(hidden_states[:, 0, :]).chunk(2, dim=-1)

    # Official Mamba.step() inserts the new input at the final convolution
    # state slot. With an otherwise-zero state, only the final kernel tap is
    # active at the first timestep.
    conv_weight = mixer.conv1d.weight[:, 0, -1].to(dtype=x_branch.dtype)
    x_branch = x_branch * conv_weight
    if mixer.conv1d.bias is not None:
        x_branch = x_branch + mixer.conv1d.bias.to(dtype=x_branch.dtype)
    x_branch = F.silu(x_branch).to(dtype=dtype)

    projected = mixer.x_proj(x_branch)
    dt_raw, b_state, c_state = torch.split(
        projected,
        [int(mixer.dt_rank), int(mixer.d_state), int(mixer.d_state)],
        dim=-1,
    )
    dt = F.softplus(mixer.dt_proj(dt_raw))

    # Starting from h_(-1)=0:
    # h_0[d,n] = x[d] * dt[d] * B[n]
    # y_0[d]   = x[d] * dt[d] * sum_n(B[n] C[n]) + D[d] * x[d]
    bc = (b_state * c_state).sum(dim=-1, keepdim=True)
    skip = mixer.D.to(dtype=x_branch.dtype)
    y = x_branch * (dt * bc + skip)
    y = y * F.silu(z_branch)
    return mixer.out_proj(y).unsqueeze(1)


class MambaARBlock(nn.Module):
    """Pre-norm residual Mamba block with optional Transformer-style MLP."""

    def __init__(self, config: MambaARConfig, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.mixer = build_mamba_mixer(config)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._length_one_fast_path_checked = False
        self._length_one_fast_path_enabled = False
        self.use_mlp = int(config.mlp_hidden_mult) > 0
        if self.use_mlp:
            self.ln_2 = nn.LayerNorm(config.d_model)
            self.mlp = MambaMLP(config)

    def _mix(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.size(1) != 1:
            return self.mixer(normalized)

        if not self._length_one_fast_path_checked:
            with torch.no_grad():
                sample = normalized[: min(4, normalized.size(0))]
                reference = self.mixer(sample)
                candidate = mamba_length_one_forward(self.mixer, sample)
                max_abs = float((reference - candidate).abs().max())
                self._length_one_fast_path_enabled = torch.allclose(
                    reference, candidate, rtol=5e-2, atol=5e-3
                )
            self._length_one_fast_path_checked = True
            is_official = self.mixer.__class__.__module__.startswith("mamba_ssm")
            if self._length_one_fast_path_enabled:
                if is_official and self.layer_idx == 0:
                    print(
                        "mamba length-1 fast path parity: PASS "
                        f"(max_abs={max_abs:.3e})",
                        flush=True,
                    )
            else:
                warnings.warn(
                    "Mamba length-1 fast path parity check failed at layer "
                    f"{self.layer_idx}; max_abs={max_abs:.3e}. Falling back to "
                    "the official selective-scan kernel.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self._length_one_fast_path_enabled:
            return mamba_length_one_forward(self.mixer, normalized)
        return self.mixer(normalized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self._mix(self.ln_1(x)))
        if self.use_mlp:
            x = x + self.mlp(self.ln_2(x))
        return x


class OthelloMambaAR(nn.Module):
    """Mamba-style autoregressive model for Othello next-move prediction."""

    def __init__(self, config: MambaARConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [MambaARBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size "
                f"{self.config.block_size}."
            )

        pos = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=TARGET_PAD,
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = MambaARConfig(
        board_size=8,
        n_layers=2,
        d_model=64,
        d_state=8,
        d_conv=3,
        expand=2,
        dropout=0.0,
        mamba_backend="torch",
    )
    model = OthelloMambaAR(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 20))
    y = torch.randint(0, cfg.vocab_size, (4, 20))
    logits, loss = model(x, y)
    assert logits.shape == (4, 20, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    print(f"OthelloMambaAR smoke passed: {model.num_parameters():,} params")
