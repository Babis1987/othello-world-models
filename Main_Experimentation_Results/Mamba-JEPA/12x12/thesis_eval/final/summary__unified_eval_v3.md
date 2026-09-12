# Unified evaluation: mamba_jepa_v5_hd_infonce_allpos_b12

- **Protocol:** `unified_eval_v3`
- **Board:** `12x12`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `1641cdb2a8f662034caaa027e93338f66e650aa227d1b84910353a41caab6466`
- **Split manifest SHA-256:** `0ef44a8ff8095d9eb17b6bd95af4ebac29288521851bdfc9edbfb580bd185e4e`
- **Data-manifest SHA-256:** `1ffe57c995aad5523da285757d3d9813a52c0eb2b1bca28294404e1f973c2c31`
- **Per-shard corpus identity:** `78ea34919eb892e91d8621569afba2ced653c7fe8b5b2c97bfdd01d369074610`
- **Project-source SHA-256:** `4fdfec5bb0841fd14c03b7c6e527d88a640c41407e6aa1f5c323d7157a9aa1ae`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 98.99% | 97.80% | 96.13% | 95.47% |
| MLP | 98.25% | 97.07% | 95.38% | 95.55% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 66.97% | 67.07% | 74.78% | 50.60% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 92.90% | 92.86% | 94.54% | 89.33% | 99.98% |
| MLP | absolute | L9 | 0.600 | 67.64% | 67.80% | 75.25% | 51.90% | 99.60% |
| MLP | relative | L12 | 0.800 | 91.87% | 91.85% | 93.77% | 87.90% | 99.88% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 63.65% | 55.76% | 65.00% | 57.43% |  |
| L1 | 69.17% | 61.48% | 73.36% | 66.94% |  |
| L2 | 67.40% | 59.71% | 77.87% | 73.44% |  |
| L3 | 67.51% | 59.80% | 78.31% | 74.06% |  |
| L4 | 67.84% | 60.11% | 79.39% | 75.41% |  |
| L5 | 72.03% | 64.19% | 83.65% | 79.44% |  |
| L6 | 73.68% | 65.87% | 85.24% | 80.97% |  |
| L7 | 74.04% | 66.24% | 85.37% | 81.02% |  |
| L8 | 74.05% | 66.26% | 87.65% | 84.00% |  |
| L9 | 74.53% | 66.80% | 87.82% | 84.14% |  |
| L10 | 74.15% | 66.35% | 93.83% | 91.98% |  |
| L11 | 74.67% | 66.94% | 94.26% | 92.50% |  |
| L12 | 74.78% | 67.07% | 94.54% | 92.86% | absolute + relative |
| L13 | 74.57% | 66.81% | 94.24% | 92.48% |  |
| L14 | 74.00% | 66.12% | 93.37% | 91.40% |  |
| L15 | 73.74% | 65.83% | 93.04% | 91.01% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.87% | 57.28% | 65.26% | 57.70% |  |
| L1 | 69.94% | 62.76% | 72.08% | 65.48% |  |
| L2 | 72.06% | 66.17% | 76.03% | 71.33% |  |
| L3 | 72.33% | 66.58% | 76.38% | 71.89% |  |
| L4 | 72.20% | 66.35% | 77.14% | 72.94% |  |
| L5 | 72.71% | 65.39% | 81.24% | 76.76% |  |
| L6 | 73.46% | 65.78% | 83.08% | 78.45% |  |
| L7 | 73.62% | 65.84% | 83.35% | 78.57% |  |
| L8 | 75.08% | 67.73% | 85.39% | 81.26% |  |
| L9 | 75.25% | 67.80% | 85.82% | 81.61% | absolute |
| L10 | 74.17% | 66.52% | 91.91% | 89.78% |  |
| L11 | 74.41% | 66.64% | 93.15% | 91.12% |  |
| L12 | 74.64% | 66.89% | 93.77% | 91.85% | relative |
| L13 | 74.33% | 66.51% | 93.20% | 91.13% |  |
| L14 | 73.59% | 65.64% | 91.89% | 89.56% |  |
| L15 | 73.20% | 65.22% | 91.41% | 89.02% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/results__unified_eval_v3.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/linear_next_move_head__unified_eval_v3.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/mlp_next_move_head__unified_eval_v3.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/linear_board_probe__unified_eval_v3.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/mlp_board_probe__unified_eval_v3.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/position_manifest__unified_eval_v3.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 2327.8 s |
| frozen_head_mlp | 2619.5 s |
| board_feature_extraction | 214.0 s |
| board_probe_linear | 40.4 s |
| board_probe_mlp | 45.7 s |
