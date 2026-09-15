# Unified evaluation: mamba_ar_b12

> **Verified v3 → v4 protocol migration (native AR only).** Metrics 
> were reused without recomputation after checkpoint, split, budget, 
> completeness, and artifact validation. The new v4 JEPA frozen-head 
> tuning stage is not applicable to a native AR head. Source result 
> SHA-256: `aaf685a3396bb1242236a7f71706b1d6d4820c258d43679e7b079dc8cf738218`.

- **Protocol:** `unified_eval_v4`
- **Board:** `12x12`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `3acd7992e6f7777a8c49d3c3acdcb331175731b3ffd0b60a711ce462618965f7`
- **Split manifest SHA-256:** `0ef44a8ff8095d9eb17b6bd95af4ebac29288521851bdfc9edbfb580bd185e4e`
- **Data-manifest SHA-256:** `1ffe57c995aad5523da285757d3d9813a52c0eb2b1bca28294404e1f973c2c31`
- **Per-shard corpus identity:** `78ea34919eb892e91d8621569afba2ced653c7fe8b5b2c97bfdd01d369074610`
- **Project-source SHA-256:** `d8e32f03a58dc21f2a5a7871756c819577c8aeea2292f2ec349fd23da4f0ce6d`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.51% | 98.27% | 96.53% | 95.41% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L10 | 0.667 | 66.82% | 66.77% | 74.55% | 50.16% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 91.39% | 91.37% | 93.41% | 87.13% | 99.97% |
| MLP | absolute | L11 | 0.733 | 66.73% | 66.68% | 74.48% | 50.03% | 99.98% |
| MLP | relative | L12 | 0.800 | 90.09% | 90.06% | 92.42% | 85.26% | 99.90% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 63.70% | 55.80% | 65.02% | 57.46% |  |
| L1 | 73.78% | 66.04% | 76.21% | 69.13% |  |
| L2 | 71.90% | 64.04% | 86.89% | 83.52% |  |
| L3 | 71.46% | 63.55% | 86.68% | 83.36% |  |
| L4 | 71.20% | 63.28% | 86.70% | 83.45% |  |
| L5 | 71.58% | 63.67% | 87.40% | 84.26% |  |
| L6 | 71.42% | 63.48% | 87.74% | 84.72% |  |
| L7 | 71.40% | 63.46% | 88.38% | 85.55% |  |
| L8 | 71.74% | 63.80% | 89.27% | 86.62% |  |
| L9 | 72.74% | 64.86% | 90.68% | 88.23% |  |
| L10 | 74.55% | 66.77% | 92.72% | 90.46% | absolute |
| L11 | 74.53% | 66.74% | 93.03% | 90.87% |  |
| L12 | 74.47% | 66.67% | 93.41% | 91.37% | relative |
| L13 | 74.44% | 66.63% | 93.38% | 91.32% |  |
| L14 | 74.28% | 66.43% | 93.20% | 91.11% |  |
| L15 | 74.26% | 66.40% | 93.19% | 91.09% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.79% | 57.14% | 65.31% | 57.74% |  |
| L1 | 73.03% | 65.29% | 74.91% | 67.60% |  |
| L2 | 70.69% | 62.80% | 84.15% | 80.59% |  |
| L3 | 70.30% | 62.41% | 83.90% | 80.46% |  |
| L4 | 69.86% | 61.95% | 83.71% | 80.30% |  |
| L5 | 70.50% | 62.57% | 84.68% | 81.34% |  |
| L6 | 70.43% | 62.48% | 84.99% | 81.74% |  |
| L7 | 70.34% | 62.37% | 85.59% | 82.53% |  |
| L8 | 70.76% | 62.82% | 86.56% | 83.69% |  |
| L9 | 72.10% | 64.21% | 88.67% | 85.98% |  |
| L10 | 74.49% | 66.69% | 91.82% | 89.30% |  |
| L11 | 74.48% | 66.68% | 92.22% | 89.80% | absolute |
| L12 | 74.39% | 66.57% | 92.42% | 90.06% | relative |
| L13 | 74.30% | 66.45% | 92.31% | 89.92% |  |
| L14 | 74.08% | 66.18% | 92.06% | 89.59% |  |
| L15 | 74.05% | 66.14% | 92.01% | 89.54% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b12/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b12/thesis_eval/final/linear_board_probe__unified_eval_v3.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b12/thesis_eval/final/mlp_board_probe__unified_eval_v3.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b12/thesis_eval/final/position_manifest__unified_eval_v3.json`

| Stage | Wall time |
|---|---:|
| native_ar | 5442.0 s |
| board_feature_extraction | 612.4 s |
| board_probe_linear | 141.3 s |
| board_probe_mlp | 161.9 s |
