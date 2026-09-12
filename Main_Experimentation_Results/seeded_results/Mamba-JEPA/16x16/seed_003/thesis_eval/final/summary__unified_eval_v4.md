# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `cc489ef82235bbaf6d70cc83217be4a11c96aaf1efb24211cf94a0376cbac53c`
- **Split manifest SHA-256:** `ae34187c52ae21615d5714279130ab2b0dc64959281854b4fceaca0a7b088cdc`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 98.37% | 97.54% | 96.51% | 90.40% |
| MLP | 98.20% | 97.36% | 96.33% | 90.15% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L13 | 0.867 | 66.30% | 66.25% | 74.34% | 49.45% | 99.85% |
| LINEAR | relative | L12 | 0.800 | 83.88% | 83.91% | 87.68% | 76.23% | 99.42% |
| MLP | absolute | L13 | 0.867 | 65.79% | 65.69% | 73.87% | 48.71% | 99.65% |
| MLP | relative | L12 | 0.800 | 81.27% | 81.31% | 85.48% | 72.80% | 98.48% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.67% | 56.53% | 65.54% | 57.69% |  |
| L1 | 66.56% | 58.40% | 67.69% | 59.88% |  |
| L2 | 69.65% | 61.52% | 71.46% | 63.89% |  |
| L3 | 69.49% | 61.37% | 71.91% | 64.55% |  |
| L4 | 70.48% | 62.40% | 74.55% | 67.79% |  |
| L5 | 69.17% | 61.07% | 77.87% | 72.72% |  |
| L6 | 71.51% | 63.39% | 80.05% | 74.76% |  |
| L7 | 73.17% | 65.10% | 81.54% | 76.16% |  |
| L8 | 73.45% | 65.39% | 81.67% | 76.25% |  |
| L9 | 73.63% | 65.57% | 81.66% | 76.17% |  |
| L10 | 73.00% | 64.90% | 86.61% | 82.85% |  |
| L11 | 73.66% | 65.59% | 87.28% | 83.55% |  |
| L12 | 74.11% | 66.03% | 87.68% | 83.91% | relative |
| L13 | 74.34% | 66.25% | 87.65% | 83.78% | absolute |
| L14 | 74.05% | 65.90% | 86.88% | 82.80% |  |
| L15 | 73.58% | 65.38% | 85.95% | 81.71% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.19% | 57.01% | 65.80% | 57.84% |  |
| L1 | 66.37% | 58.26% | 67.15% | 59.28% |  |
| L2 | 68.47% | 60.41% | 69.82% | 62.11% |  |
| L3 | 68.20% | 60.15% | 69.87% | 62.29% |  |
| L4 | 68.86% | 60.84% | 71.92% | 64.88% |  |
| L5 | 68.86% | 61.19% | 74.94% | 69.47% |  |
| L6 | 70.10% | 62.08% | 76.88% | 71.27% |  |
| L7 | 71.95% | 63.89% | 78.76% | 72.97% |  |
| L8 | 72.42% | 64.37% | 79.03% | 73.14% |  |
| L9 | 72.65% | 64.57% | 79.09% | 73.13% |  |
| L10 | 71.79% | 63.66% | 83.66% | 79.58% |  |
| L11 | 72.72% | 64.60% | 84.79% | 80.70% |  |
| L12 | 73.48% | 65.34% | 85.48% | 81.31% | relative |
| L13 | 73.87% | 65.69% | 85.59% | 81.23% | absolute |
| L14 | 73.53% | 65.29% | 84.54% | 79.89% |  |
| L15 | 73.02% | 64.72% | 83.52% | 78.70% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b16_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed003/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed003/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b16_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 21812.2 s |
| head_tuning_linear | 1672.6 s |
| head_tuning_mlp | 1705.7 s |
| frozen_head_mlp | 22472.3 s |
| board_feature_extraction | 1744.5 s |
| board_probe_linear | 172.2 s |
| board_probe_mlp | 191.7 s |
