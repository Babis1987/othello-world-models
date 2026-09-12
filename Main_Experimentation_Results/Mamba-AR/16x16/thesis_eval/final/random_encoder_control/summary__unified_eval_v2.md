# Unified evaluation: mamba_ar_b16__random_encoder_control

- **Protocol:** `unified_eval_v2`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `architecture_matched_random_encoder_control`
- **Checkpoint:** `final.pt` (step `0`, games `0`)
- **Checkpoint SHA-256:** `511e0a62b5a70110268ff026c5e266c9624e1615a73aa6f1e3514f70be487699`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `f6d85c1fed548dc383e645efa0eed0c313d0efcd33b922e2b62451c603f2d826`
- **Exact pretraining budget:** `0` games / `0` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts (primary comparison)

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L0 | 0.000 | 55.53% | 55.35% | 63.54% | 38.34% | 89.38% |
| LINEAR | relative | L0 | 0.000 | 56.65% | 56.40% | 64.33% | 40.04% | 89.24% |
| MLP | absolute | L0 | 0.000 | 56.48% | 56.33% | 64.53% | 39.29% | 90.42% |
| MLP | relative | L0 | 0.000 | 57.23% | 57.16% | 65.21% | 40.71% | 90.34% |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/random_encoder_control/results__unified_eval_v2.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/random_encoder_control/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/random_encoder_control/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/random_encoder_control/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 924.0 s |
| board_probe_linear | 6.8 s |
| board_probe_mlp | 8.6 s |
