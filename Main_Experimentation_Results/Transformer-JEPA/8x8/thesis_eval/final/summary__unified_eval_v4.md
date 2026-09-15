# Unified evaluation: jepa_v5_infonce_b8_run_001_hd_allpos_bf16

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `0c4eda770574ddd8577a7bf7ca0422a659e374232226e8b8ea7359c7d3447001`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.42% | 96.55% | 92.26% | 97.76% |
| MLP | 99.42% | 96.53% | 92.23% | 97.58% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 69.03% | 69.02% | 75.69% | 53.58% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.44% | 96.72% | 97.45% | 95.15% | 99.99% |
| MLP | absolute | L5 | 0.625 | 94.70% | 94.83% | 95.94% | 92.25% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.14% | 96.41% | 97.22% | 94.72% | 99.97% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.70% | 55.78% | 64.81% | 58.17% |  |
| L1 | 75.09% | 68.39% | 87.25% | 83.74% |  |
| L2 | 75.68% | 69.02% | 91.54% | 89.13% |  |
| L3 | 75.69% | 69.03% | 94.04% | 92.35% |  |
| L4 | 75.69% | 69.02% | 95.99% | 94.85% | absolute |
| L5 | 75.68% | 69.01% | 97.17% | 96.37% |  |
| L6 | 75.55% | 68.85% | 97.45% | 96.72% | relative |
| L7 | 75.21% | 68.50% | 96.97% | 96.16% |  |
| L8 | 75.03% | 68.34% | 96.60% | 95.72% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.12% | 58.78% | 65.24% | 58.63% |  |
| L1 | 84.18% | 80.06% | 86.50% | 82.85% |  |
| L2 | 88.56% | 85.43% | 91.24% | 88.74% |  |
| L3 | 91.73% | 89.47% | 93.82% | 92.07% |  |
| L4 | 94.07% | 92.45% | 95.81% | 94.62% |  |
| L5 | 95.94% | 94.83% | 97.03% | 96.18% | absolute |
| L6 | 94.64% | 93.19% | 97.22% | 96.41% | relative |
| L7 | 92.52% | 90.61% | 96.48% | 95.55% |  |
| L8 | 89.53% | 86.89% | 95.97% | 94.97% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 90.90% | 88.84% | 2.416 |
| Random direction | 92.20% | 88.65% | 2.420 |
| Relative-board direction | 95.40% | 93.30% | 0.598 |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 145.4 s |
| board_probe_linear | 54.4 s |
| board_probe_mlp | 65.6 s |
| causal_intervention | 68.0 s |
| frozen_head_linear | 1660.1 s |
| frozen_head_mlp | 1484.8 s |
| head_tuning_linear | 99.5 s |
| head_tuning_mlp | 2.1 s |
