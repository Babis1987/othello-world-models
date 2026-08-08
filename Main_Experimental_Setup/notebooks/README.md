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
  one-batch gate succeeds.
- A model factory is defined visibly in each notebook and is invoked exactly
  once by the trainer after the seed is set. There is no preview-model
  construction that can move the global Torch RNG.
- The full configuration is resolved from the locked files in `../configs`.
- The four notebooks do not redefine training math, checkpointing, metrics,
  progress output, or Drive synchronization.
- The official Mamba notebooks pin `mamba-ssm==2.3.2.post1` and
  `causal-conv1d==1.6.2.post1`.

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
