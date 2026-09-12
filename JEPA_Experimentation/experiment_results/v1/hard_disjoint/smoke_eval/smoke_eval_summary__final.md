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
| Linear | 73.11% | 65.26% | 58.81% | 42.53% |
| MLP | 78.93% | 70.97% | 63.90% | 47.49% |

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L8 | 64.19% | L2 | 66.63% |
| MLP | L2 | 64.85% | L2 | 67.12% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 59.92% | 61.74% |
| L1 | 62.92% | 66.16% |
| L2 | 63.54% | 66.63% |
| L3 | 63.05% | 65.97% |
| L4 | 63.68% | 66.41% |
| L5 | 63.69% | 66.24% |
| L6 | 64.02% | 66.19% |
| L7 | 63.94% | 66.52% |
| L8 | 64.19% | 65.78% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 61.61% | 63.17% |
| L1 | 64.29% | 66.96% |
| L2 | 64.85% | 67.12% |
| L3 | 64.54% | 66.86% |
| L4 | 64.64% | 66.41% |
| L5 | 63.98% | 66.52% |
| L6 | 63.56% | 65.12% |
| L7 | 63.16% | 64.33% |
| L8 | 62.42% | 63.64% |

## Matched-Conditions Board Probes (pass-free, last position, t in [4,44])

Probes trained only on hidden states at position t-1 of pass-free prefixes — the exact conditions prefix-based objectives supervise. Compare against the all-position tables above.

| Layer | Linear abs | Linear rel | MLP abs | MLP rel |
|---:|---:|---:|---:|---:|
| L0 | 65.26% | 66.97% | 65.15% | 66.90% |
| L1 | 65.63% | 68.42% | 68.00% | 70.46% |
| L2 | 67.00% | 69.63% | 68.92% | 71.39% |
| L3 | 67.66% | 69.81% | 68.88% | 71.40% |
| L4 | 67.50% | 70.48% | 69.28% | 71.66% |
| L5 | 67.30% | 69.33% | 69.42% | 71.67% |
| L6 | 67.87% | 70.08% | 69.10% | 70.49% |
| L7 | 68.02% | 68.74% | 68.38% | 69.45% |
| L8 | 67.20% | 69.09% | 67.26% | 68.68% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_next_move_head__final.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_next_move_head__final.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_board_probes__final.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_board_probes__final.pt`
