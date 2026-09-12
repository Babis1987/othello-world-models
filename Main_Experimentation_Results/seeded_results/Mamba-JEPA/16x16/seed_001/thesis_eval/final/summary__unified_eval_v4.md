# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `3bdf5631e3bc6c89754742c33cbfe961436e4a01fec60377d37a2ccbc444b771`
- **Split manifest SHA-256:** `6ab3d73bb5d53d60705c8eedb9df71b11db5be0c7eedef854017acd2fabe93fb`
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
| LINEAR | 98.41% | 97.59% | 96.57% | 90.70% |
| MLP | 98.23% | 97.40% | 96.37% | 90.43% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L9 | 0.600 | 66.34% | 66.24% | 74.29% | 49.52% | 99.67% |
| LINEAR | relative | L11 | 0.733 | 83.85% | 83.81% | 87.57% | 76.15% | 99.28% |
| MLP | absolute | L9 | 0.600 | 65.63% | 65.55% | 73.65% | 48.71% | 99.21% |
| MLP | relative | L12 | 0.800 | 81.07% | 81.03% | 85.30% | 72.41% | 98.50% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.68% | 56.53% | 65.57% | 57.69% |  |
| L1 | 66.53% | 58.46% | 67.65% | 59.92% |  |
| L2 | 66.17% | 58.09% | 68.93% | 61.82% |  |
| L3 | 67.58% | 59.46% | 70.84% | 63.85% |  |
| L4 | 72.65% | 64.58% | 75.53% | 68.36% |  |
| L5 | 72.29% | 64.19% | 80.74% | 75.40% |  |
| L6 | 73.35% | 65.26% | 81.71% | 76.31% |  |
| L7 | 73.84% | 65.76% | 82.06% | 76.60% |  |
| L8 | 74.05% | 65.98% | 82.24% | 76.77% |  |
| L9 | 74.29% | 66.24% | 82.22% | 76.68% | absolute |
| L10 | 73.71% | 65.58% | 87.27% | 83.46% |  |
| L11 | 73.87% | 65.75% | 87.57% | 83.81% | relative |
| L12 | 74.10% | 65.98% | 87.55% | 83.72% |  |
| L13 | 74.11% | 65.98% | 87.09% | 83.10% |  |
| L14 | 73.92% | 65.76% | 86.37% | 82.18% |  |
| L15 | 73.57% | 65.38% | 85.59% | 81.27% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.20% | 57.02% | 65.80% | 57.84% |  |
| L1 | 66.60% | 58.54% | 67.29% | 59.41% |  |
| L2 | 66.44% | 58.50% | 68.05% | 60.68% |  |
| L3 | 67.06% | 59.10% | 69.26% | 62.05% |  |
| L4 | 71.51% | 63.45% | 73.60% | 66.17% |  |
| L5 | 71.14% | 63.09% | 77.91% | 72.27% |  |
| L6 | 72.16% | 64.05% | 78.96% | 73.14% |  |
| L7 | 72.87% | 64.76% | 79.53% | 73.60% |  |
| L8 | 73.24% | 65.12% | 79.87% | 73.90% |  |
| L9 | 73.65% | 65.55% | 80.06% | 74.00% | absolute |
| L10 | 72.79% | 64.61% | 84.55% | 80.35% |  |
| L11 | 73.05% | 64.88% | 85.19% | 81.07% |  |
| L12 | 73.44% | 65.26% | 85.30% | 81.03% | relative |
| L13 | 73.58% | 65.37% | 84.84% | 80.33% |  |
| L14 | 73.39% | 65.14% | 83.91% | 79.10% |  |
| L15 | 72.99% | 64.70% | 83.06% | 78.09% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b16_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed001/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed001/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b16_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1757.8 s |
| board_probe_linear | 177.6 s |
| board_probe_mlp | 205.1 s |
| frozen_head_linear | 20490.9 s |
| frozen_head_mlp | 19590.6 s |
| head_tuning_linear | 1577.7 s |
| head_tuning_mlp | 10.3 s |
