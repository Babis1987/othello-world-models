# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `325876cd5b57f99214b75b95944ab63a33a8ca161649e5ac930b9171657c8744`
- **Split manifest SHA-256:** `7eb75adbb502c800ca8da8359cf6cf48b8f7e12e1b9e3bea576efdbf45ab51b6`
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
| LINEAR | 98.55% | 97.74% | 96.72% | 90.71% |
| MLP | 98.41% | 97.59% | 96.58% | 90.43% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L13 | 0.867 | 66.20% | 66.18% | 74.28% | 49.36% | 99.81% |
| LINEAR | relative | L13 | 0.867 | 84.09% | 84.11% | 87.88% | 76.43% | 99.61% |
| MLP | absolute | L13 | 0.867 | 65.66% | 65.63% | 73.80% | 48.66% | 99.56% |
| MLP | relative | L13 | 0.867 | 81.61% | 81.62% | 85.84% | 73.05% | 98.95% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.66% | 56.50% | 65.56% | 57.68% |  |
| L1 | 66.90% | 58.80% | 68.46% | 60.86% |  |
| L2 | 67.38% | 59.29% | 71.23% | 64.47% |  |
| L3 | 69.06% | 60.97% | 72.68% | 65.79% |  |
| L4 | 69.77% | 61.70% | 72.91% | 65.86% |  |
| L5 | 69.45% | 61.40% | 74.00% | 67.47% |  |
| L6 | 68.69% | 60.62% | 78.54% | 73.72% |  |
| L7 | 67.67% | 59.54% | 83.14% | 79.63% |  |
| L8 | 69.37% | 61.28% | 84.17% | 80.66% |  |
| L9 | 70.98% | 62.89% | 85.40% | 81.87% |  |
| L10 | 72.42% | 64.31% | 86.66% | 83.10% |  |
| L11 | 73.49% | 65.41% | 87.52% | 83.92% |  |
| L12 | 73.99% | 65.90% | 87.79% | 84.10% |  |
| L13 | 74.28% | 66.18% | 87.88% | 84.11% | absolute + relative |
| L14 | 73.95% | 65.78% | 87.07% | 83.09% |  |
| L15 | 73.46% | 65.25% | 86.12% | 82.00% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.22% | 57.05% | 65.82% | 57.87% |  |
| L1 | 66.67% | 58.65% | 67.67% | 59.96% |  |
| L2 | 66.85% | 58.87% | 69.69% | 62.73% |  |
| L3 | 67.97% | 59.99% | 70.79% | 63.73% |  |
| L4 | 68.67% | 60.64% | 71.10% | 63.85% |  |
| L5 | 68.77% | 60.84% | 72.17% | 65.41% |  |
| L6 | 70.13% | 62.93% | 76.11% | 71.02% |  |
| L7 | 67.20% | 59.20% | 80.54% | 76.75% |  |
| L8 | 68.32% | 60.22% | 81.54% | 77.81% |  |
| L9 | 69.89% | 61.79% | 82.78% | 79.02% |  |
| L10 | 71.39% | 63.27% | 84.12% | 80.28% |  |
| L11 | 72.70% | 64.59% | 85.29% | 81.42% |  |
| L12 | 73.36% | 65.21% | 85.64% | 81.59% |  |
| L13 | 73.80% | 65.63% | 85.84% | 81.62% | absolute + relative |
| L14 | 73.43% | 65.18% | 84.77% | 80.24% |  |
| L15 | 72.87% | 64.58% | 83.74% | 79.07% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b16_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed002/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed002/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b16_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1713.6 s |
| board_probe_linear | 179.0 s |
| board_probe_mlp | 198.3 s |
| frozen_head_linear | 21607.1 s |
| frozen_head_mlp | 20272.0 s |
| head_tuning_linear | 1679.8 s |
| head_tuning_mlp | 10.6 s |
