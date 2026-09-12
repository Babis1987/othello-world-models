# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `2da5d0c983cd5cca2e0b677249a6239101e4470e2602fca80b01c30637746fc0`
- **Split manifest SHA-256:** `2a05ae6269be0a5978ba8165d947ad3098c24111b4834db50e05150a7f7845dc`
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
| LINEAR | 98.53% | 97.74% | 96.73% | 91.02% |
| MLP | 98.38% | 97.58% | 96.57% | 90.74% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L13 | 0.867 | 66.14% | 66.08% | 74.20% | 49.24% | 99.78% |
| LINEAR | relative | L11 | 0.733 | 83.88% | 83.87% | 87.52% | 76.41% | 98.91% |
| MLP | absolute | L13 | 0.867 | 65.58% | 65.53% | 73.72% | 48.51% | 99.56% |
| MLP | relative | L13 | 0.867 | 81.20% | 81.20% | 85.52% | 72.41% | 98.96% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.61% | 56.45% | 65.54% | 57.68% |  |
| L1 | 70.81% | 62.64% | 71.83% | 63.93% |  |
| L2 | 70.29% | 62.17% | 71.39% | 63.59% |  |
| L3 | 68.78% | 60.68% | 75.82% | 70.12% |  |
| L4 | 68.86% | 60.75% | 75.56% | 69.73% |  |
| L5 | 70.56% | 62.45% | 76.96% | 70.99% |  |
| L6 | 72.11% | 64.04% | 78.14% | 72.02% |  |
| L7 | 70.63% | 62.50% | 85.38% | 81.75% |  |
| L8 | 71.58% | 63.48% | 85.90% | 82.25% |  |
| L9 | 72.50% | 64.39% | 86.59% | 82.94% |  |
| L10 | 73.09% | 65.00% | 87.09% | 83.46% |  |
| L11 | 73.60% | 65.52% | 87.52% | 83.87% | relative |
| L12 | 73.89% | 65.79% | 87.63% | 83.90% |  |
| L13 | 74.20% | 66.08% | 87.67% | 83.85% | absolute |
| L14 | 73.91% | 65.74% | 86.91% | 82.89% |  |
| L15 | 73.48% | 65.28% | 86.04% | 81.88% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.20% | 57.03% | 65.82% | 57.88% |  |
| L1 | 69.27% | 61.17% | 70.05% | 62.09% |  |
| L2 | 68.90% | 60.82% | 69.74% | 61.81% |  |
| L3 | 68.24% | 60.34% | 73.52% | 67.51% |  |
| L4 | 68.06% | 60.08% | 73.19% | 67.08% |  |
| L5 | 69.34% | 61.34% | 74.39% | 68.12% |  |
| L6 | 70.93% | 62.89% | 75.64% | 69.17% |  |
| L7 | 69.25% | 61.16% | 82.20% | 78.30% |  |
| L8 | 70.28% | 62.19% | 82.86% | 78.92% |  |
| L9 | 71.33% | 63.21% | 83.78% | 79.84% |  |
| L10 | 72.08% | 64.00% | 84.52% | 80.56% |  |
| L11 | 72.81% | 64.69% | 85.14% | 81.10% |  |
| L12 | 73.25% | 65.10% | 85.33% | 81.19% |  |
| L13 | 73.72% | 65.53% | 85.52% | 81.20% | absolute + relative |
| L14 | 73.40% | 65.14% | 84.47% | 79.84% |  |
| L15 | 72.90% | 64.62% | 83.53% | 78.73% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b16_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed000/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b16_seed000/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b16_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b16_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1663.1 s |
| board_probe_linear | 177.8 s |
| board_probe_mlp | 199.6 s |
| frozen_head_linear | 20722.6 s |
| frozen_head_mlp | 19516.2 s |
| head_tuning_linear | 1589.9 s |
| head_tuning_mlp | 10.4 s |
