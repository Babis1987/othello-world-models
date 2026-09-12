# JEPA Smoke Evaluation: jepa_vicreg_b8_run_001_hard_disjoint_bf16

- **Checkpoint:** `final.pt`
- **Objective:** `jepa`
- **Checkpoint step:** `78200`
- **Games seen:** `19,999,840`
- **Downstream training chunks:** `20`
- **Board probe training games:** `5,000` (`250` per chunk)
- **Validation split begins after pretraining chunk:** `200`

## Next-Move Heads: Legal-Move Evaluation

The heads are trained with next-move cross-entropy. These metrics evaluate whether their predictions are legal, not whether they match the sampled continuation.

| Head | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal probability mass |
|---|---:|---:|---:|---:|
| Linear | 73.25% | 65.35% | 58.87% | 42.66% |
| MLP | 79.11% | 71.05% | 63.86% | 47.58% |

- **Board probe protocol:** `normalized_v2` (input LayerNorm=True, precision=bf16)

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L7 | 66.78% | L5 | 70.29% |
| MLP | L8 | 66.58% | L4 | 69.82% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 62.74% | 65.00% |
| L1 | 64.77% | 68.68% |
| L2 | 65.84% | 69.81% |
| L3 | 66.06% | 69.87% |
| L4 | 66.53% | 70.27% |
| L5 | 66.68% | 70.29% |
| L6 | 66.69% | 70.15% |
| L7 | 66.78% | 70.08% |
| L8 | 66.69% | 69.84% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 64.70% | 65.14% |
| L1 | 65.71% | 68.76% |
| L2 | 66.35% | 69.50% |
| L3 | 66.23% | 69.51% |
| L4 | 66.44% | 69.82% |
| L5 | 66.55% | 69.78% |
| L6 | 66.46% | 69.71% |
| L7 | 66.49% | 69.62% |
| L8 | 66.58% | 69.41% |

## Matched-Conditions Board Probes (pass-free, last position, t in [4,44])

Probes trained only on hidden states at position t-1 of pass-free prefixes — the exact conditions prefix-based objectives supervise. Compare against the all-position tables above.

| Layer | Linear abs | Linear rel | MLP abs | MLP rel |
|---:|---:|---:|---:|---:|
| L0 | 64.88% | 67.06% | 67.43% | 67.60% |
| L1 | 68.64% | 72.09% | 70.66% | 72.57% |
| L2 | 69.91% | 73.75% | 71.43% | 73.93% |
| L3 | 70.27% | 73.93% | 71.44% | 73.98% |
| L4 | 71.03% | 74.58% | 71.85% | 74.52% |
| L5 | 71.39% | 74.75% | 71.53% | 74.55% |
| L6 | 71.43% | 74.72% | 71.33% | 74.59% |
| L7 | 71.47% | 74.72% | 71.36% | 74.43% |
| L8 | 71.54% | 74.55% | 71.27% | 74.31% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_next_move_head__final.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_next_move_head__final.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_board_probes_normalized_v2__final.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_board_probes_normalized_v2__final.pt`
