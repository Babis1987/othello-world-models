# Canonical training notebooks

This directory contains one notebook per cell of the architecture-objective
comparison. The architecture and objective are fixed by the filename; change
only `BOARD_SIZE` at the top to select 8, 12, or 16.

| Notebook | Fixed experiment |
|---|---|
| `train_transformer_ar.ipynb` | Transformer + autoregressive objective |
| `train_mamba_ar.ipynb` | Mamba + autoregressive objective |
| `train_transformer_jepa.ipynb` | Transformer + final all-position JEPA |
| `train_mamba_jepa.ipynb` | Mamba + final all-position JEPA |

## Safety contract

- `BOARD_SIZE` defaults to 12.
- `RUN_PROFILE` defaults to `"smoke"`; change it to `"full"` only after the
  smoke gate succeeds: one input shard for AR and one batch for JEPA.
- A model factory is defined visibly in each notebook and is invoked exactly
  once by the trainer after the seed is set. There is no preview-model
  construction that can move the global Torch RNG.
- The full configuration is resolved from the locked files in `../configs`.
- The four notebooks do not redefine training math, checkpointing, metrics,
  progress output, or Drive synchronization.
- The official Mamba notebooks pin `mamba-ssm==2.3.2.post1` and
  `causal-conv1d==1.6.2.post1`.

The final production-backend gate is stricter than the notebooks' package
metadata check. It passed in an isolated Colab runtime using the successor
source environment lock's Torch `2.11.0+cu128` / CUDA `12.8` software stack.
To repeat it, run `python tools/smoke_official_mamba.py` from the repository
root and require `OFFICIAL_MAMBA_SMOKE_PASS`. The script constructs the exact
board-12 geometry and executes one no-write BF16 optimizer step. It also checks
all 15 length-one layers and proves that a forced parity warning cannot disable
the resume-stable fast path.

## Colab path

The repository is expected at:

```text
/content/drive/Othercomputers/MyLaptop/Master_Thesis_Code_Final
```

Training artifacts remain under:

```text
/content/drive/MyDrive/Master_Thesis_Artifacts
```

The notebooks reject an incomplete corpus manifest before constructing a
model. Resume uses the selected run's synced `latest.pt`; because the original
checkpoint format does not store Python/Torch/CUDA RNG states, a resumed run is
operationally supported but is not guaranteed to be bitwise identical to an
uninterrupted run.
