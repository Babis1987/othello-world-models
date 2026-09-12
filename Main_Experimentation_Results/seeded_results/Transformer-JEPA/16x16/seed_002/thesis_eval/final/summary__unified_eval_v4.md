# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `bbf3750b24981667030cde758d79aba5e90ca6ab74cb7c6fb253135ecfd60bf5`
- **Split manifest SHA-256:** `ed9780f857eeb4c462b85426f24263937f6902d2c0525102423ff56ca49aeeb1`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 97.51% | 96.47% | 95.28% | 87.97% |
| MLP | 97.16% | 96.12% | 94.93% | 87.67% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L6 | 0.750 | 66.33% | 66.30% | 74.39% | 49.51% | 99.89% |
| LINEAR | relative | L6 | 0.750 | 78.45% | 78.54% | 83.69% | 68.00% | 99.77% |
| MLP | absolute | L6 | 0.750 | 65.81% | 65.81% | 73.92% | 48.95% | 99.51% |
| MLP | relative | L6 | 0.750 | 76.15% | 76.18% | 81.73% | 64.85% | 99.02% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.55% | 56.49% | 65.42% | 57.63% |  |
| L1 | 65.66% | 57.62% | 67.00% | 59.38% |  |
| L2 | 68.22% | 60.17% | 69.54% | 61.89% |  |
| L3 | 70.64% | 62.57% | 72.13% | 64.51% |  |
| L4 | 72.52% | 64.43% | 74.37% | 66.82% |  |
| L5 | 73.91% | 65.80% | 80.49% | 74.48% |  |
| L6 | 74.39% | 66.30% | 83.69% | 78.54% | absolute + relative |
| L7 | 74.03% | 65.91% | 83.43% | 78.32% |  |
| L8 | 73.69% | 65.60% | 82.44% | 77.17% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.52% | 57.50% | 65.76% | 57.80% |  |
| L1 | 66.07% | 57.99% | 66.62% | 58.69% |  |
| L2 | 67.58% | 59.52% | 68.31% | 60.41% |  |
| L3 | 69.39% | 61.38% | 70.34% | 62.54% |  |
| L4 | 70.97% | 62.88% | 72.24% | 64.50% |  |
| L5 | 72.71% | 64.57% | 77.98% | 71.57% |  |
| L6 | 73.92% | 65.81% | 81.73% | 76.18% | absolute + relative |
| L7 | 73.56% | 65.36% | 81.49% | 75.90% |  |
| L8 | 73.08% | 64.89% | 80.14% | 74.30% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b16_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed002/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed002/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b16_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1655.4 s |
| board_probe_linear | 108.9 s |
| board_probe_mlp | 121.8 s |
| frozen_head_linear | 20836.8 s |
| frozen_head_mlp | 19615.8 s |
| head_tuning_linear | 1594.0 s |
| head_tuning_mlp | 10.4 s |
