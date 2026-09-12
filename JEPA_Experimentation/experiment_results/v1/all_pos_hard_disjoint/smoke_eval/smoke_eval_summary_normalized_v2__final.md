# JEPA Smoke Evaluation: jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16

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
| Linear | 99.00% | 96.01% | 91.42% | 86.81% |
| MLP | 98.27% | 95.38% | 90.96% | 91.36% |

- **Board probe protocol:** `normalized_v2` (input LayerNorm=True, precision=bf16)

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L2 | 75.42% | L3 | 92.53% |
| MLP | L2 | 86.07% | L2 | 92.14% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 62.89% | 64.90% |
| L1 | 75.37% | 89.18% |
| L2 | 75.42% | 92.51% |
| L3 | 74.81% | 92.53% |
| L4 | 72.29% | 87.41% |
| L5 | 71.51% | 86.06% |
| L6 | 70.93% | 84.46% |
| L7 | 70.27% | 82.70% |
| L8 | 69.76% | 80.92% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 64.99% | 65.17% |
| L1 | 84.57% | 88.67% |
| L2 | 86.07% | 92.14% |
| L3 | 77.52% | 91.82% |
| L4 | 70.90% | 84.36% |
| L5 | 70.11% | 82.21% |
| L6 | 69.58% | 80.73% |
| L7 | 68.96% | 78.93% |
| L8 | 68.49% | 77.31% |

## Matched-Conditions Board Probes (pass-free, last position, t in [4,44])

Probes trained only on hidden states at position t-1 of pass-free prefixes — the exact conditions prefix-based objectives supervise. Compare against the all-position tables above.

| Layer | Linear abs | Linear rel | MLP abs | MLP rel |
|---:|---:|---:|---:|---:|
| L0 | 64.91% | 66.92% | 67.40% | 67.62% |
| L1 | 79.50% | 92.41% | 88.56% | 91.90% |
| L2 | 79.61% | 95.61% | 92.00% | 95.28% |
| L3 | 79.15% | 95.84% | 86.10% | 95.20% |
| L4 | 77.49% | 92.22% | 77.25% | 90.02% |
| L5 | 76.76% | 91.03% | 76.44% | 88.59% |
| L6 | 76.07% | 89.44% | 75.55% | 86.82% |
| L7 | 75.28% | 87.73% | 74.74% | 85.00% |
| L8 | 74.66% | 86.06% | 73.84% | 83.29% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/smoke_eval/linear_next_move_head__final.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/smoke_eval/mlp_next_move_head__final.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/smoke_eval/linear_board_probes_normalized_v2__final.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/smoke_eval/mlp_board_probes_normalized_v2__final.pt`
