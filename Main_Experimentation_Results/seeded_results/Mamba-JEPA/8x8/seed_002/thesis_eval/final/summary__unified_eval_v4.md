# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `2d233f6c79f9a072f9e942a4ef8e831ec389ba3aed4236291a01efe0eac412c2`
- **Split manifest SHA-256:** `5a02b605879dcfdb5ad7c10ecf2440cb0363a46e15091410360ef59697de2900`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.74% | 96.82% | 92.49% | 98.53% |
| MLP | 99.73% | 96.80% | 92.48% | 98.97% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 69.10% | 69.09% | 75.74% | 53.68% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 97.94% | 98.19% | 98.60% | 97.32% | 100.00% |
| MLP | absolute | L11 | 0.733 | 94.04% | 94.16% | 95.41% | 91.24% | 100.00% |
| MLP | relative | L12 | 0.800 | 97.74% | 97.99% | 98.44% | 97.03% | 99.99% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.60% | 55.55% | 64.82% | 58.18% |  |
| L1 | 72.12% | 65.24% | 78.12% | 72.72% |  |
| L2 | 74.07% | 67.20% | 83.12% | 78.60% |  |
| L3 | 74.87% | 68.06% | 84.45% | 80.14% |  |
| L4 | 74.00% | 67.12% | 91.55% | 89.34% |  |
| L5 | 73.82% | 66.90% | 92.24% | 90.22% |  |
| L6 | 73.80% | 66.88% | 92.67% | 90.78% |  |
| L7 | 73.98% | 67.07% | 93.05% | 91.26% |  |
| L8 | 73.06% | 66.05% | 96.24% | 95.32% |  |
| L9 | 74.58% | 67.74% | 97.05% | 96.27% |  |
| L10 | 75.31% | 68.57% | 97.72% | 97.07% |  |
| L11 | 75.59% | 68.90% | 98.12% | 97.58% |  |
| L12 | 75.74% | 69.09% | 98.60% | 98.19% | absolute + relative |
| L13 | 75.59% | 68.90% | 98.44% | 97.99% |  |
| L14 | 74.87% | 68.07% | 97.16% | 96.43% |  |
| L15 | 74.72% | 67.96% | 96.38% | 95.50% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.06% | 58.71% | 65.27% | 58.68% |  |
| L1 | 75.25% | 69.36% | 77.45% | 71.95% |  |
| L2 | 78.83% | 73.40% | 82.10% | 77.42% |  |
| L3 | 78.75% | 73.05% | 83.34% | 78.80% |  |
| L4 | 87.28% | 84.19% | 90.56% | 88.19% |  |
| L5 | 87.87% | 84.99% | 91.19% | 89.03% |  |
| L6 | 86.19% | 82.80% | 91.53% | 89.48% |  |
| L7 | 87.33% | 84.22% | 91.90% | 89.93% |  |
| L8 | 92.69% | 91.33% | 95.21% | 94.22% |  |
| L9 | 92.16% | 90.27% | 96.44% | 95.59% |  |
| L10 | 93.38% | 91.63% | 97.48% | 96.77% |  |
| L11 | 95.41% | 94.16% | 97.97% | 97.39% | absolute |
| L12 | 94.54% | 93.06% | 98.44% | 97.99% | relative |
| L13 | 86.73% | 83.12% | 98.14% | 97.62% |  |
| L14 | 83.27% | 78.82% | 96.47% | 95.58% |  |
| L15 | 78.87% | 73.31% | 95.46% | 94.38% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 90.50% | 90.25% | 2.364 |
| Random direction | 91.30% | 90.03% | 2.338 |
| Relative-board direction | 96.00% | 91.60% | 1.082 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b8_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed002/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed002/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b8_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 193.6 s |
| board_probe_linear | 102.5 s |
| board_probe_mlp | 122.4 s |
| causal_intervention | 111.4 s |
| frozen_head_linear | 2111.1 s |
| frozen_head_mlp | 2952.8 s |
| head_tuning_linear | 94.5 s |
| head_tuning_mlp | 2.2 s |
