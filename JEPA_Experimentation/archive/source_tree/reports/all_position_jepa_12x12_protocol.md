# 12x12 v1 VICReg JEPA all-position ablation

This is a separate ablation. It does not modify the selected historical v1
random-boundary runs.

## Objective

- `objective_class: jepa`
- `variant: v1`
- `loss_type: vicreg`
- `view_mode: hard_disjoint_future`
- `prediction_horizon: 1`
- `position_sampling: all`

The context encoder is causal and runs once over the complete input sequence.
Every context hidden state at boundary `t` predicts the independently encoded
length-one target move at `t+1`. The target never receives the prefix.

VICReg statistics are position-stratified: variance and covariance are
computed across games at the same absolute position, then pair-count weighted
across positions. This prevents absolute position embeddings from supplying a
trivial source of variance.

## Configs

- Transformer: `configs/jepa_v1_vicreg_b12_hard_disjoint_all_positions.yml`
- Mamba: `configs/mamba_jepa_v1_hd_vicreg_b12_all_positions.yml`

Both keep the selected architecture, bf16, batch size 256, seed 42, optimiser,
and 20M-game split unchanged.

## Colab smoke gates

```bash
python -u scripts/launch_12x12_training.py transformer_jepa_allpos --smoke
python -u scripts/launch_12x12_training.py mamba_jepa_allpos --smoke
```

Use the smoke chunk's `dt` to compare against the corresponding historical-v1
smoke on the same GPU. Do not extrapolate runtime across different GPU types.

## Full runs

```bash
python -u scripts/launch_12x12_training.py transformer_jepa_allpos
python -u scripts/launch_12x12_training.py mamba_jepa_allpos
```

Expected Drive outputs:

- `runs/Transformer-JEPA/transformer_jepa_v1_hd_vicreg_allpos_b12`
- `runs/Mamba-JEPA/mamba_jepa_v1_hd_vicreg_allpos_b12`

The all-position training log includes `pos/game=...`. On near-terminal 12x12
games this should approach 138, the full set of boundaries available in the
JEPA input tensor.

## Interpretation

This run is not a drop-in replacement for the original v1 experiment. It is a
supervision-density ablation. Main cross-board comparisons should not mix
random-boundary 8x8 checkpoints with all-position 12x12/16x16 checkpoints.
