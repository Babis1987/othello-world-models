# Thesis 2×2 factorial comparison — 8x8

- **Suite:** `thesis_eval_suite_v1` / `unified_eval_v4`
- **Split:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Training precision:** `transformer_ar=bf16`, `transformer_jepa=bf16`, `mamba_ar=bf16`, `mamba_jepa=bf16`

| Metric | Transformer AR | Transformer JEPA | Mamba AR | Mamba JEPA | Architecture effect | Objective effect | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_board_absolute | 68.88% | 69.02% | 68.78% | 68.92% | -0.10% | 0.14% | -0.01% |
| linear_board_relative | 96.91% | 96.72% | 97.85% | 98.04% | 1.12% | 0.00% | 0.38% |
| linear_legal_lift | 99.68% | 99.32% | 99.82% | 99.67% | 0.25% | -0.25% | 0.21% |
| linear_legal_mass | 98.48% | 97.76% | 99.34% | 98.61% | 0.86% | -0.72% | -0.02% |
| linear_legal_top1 | 99.73% | 99.42% | 99.85% | 99.72% | 0.21% | -0.22% | 0.18% |
| mlp_board_absolute | 93.88% | 94.83% | 86.55% | 89.31% | -6.43% | 1.85% | 1.81% |
| mlp_board_relative | 96.61% | 96.41% | 97.62% | 97.82% | 1.20% | -0.00% | 0.40% |
| mlp_legal_lift | 99.68% | 99.32% | 99.82% | 99.67% | 0.25% | -0.25% | 0.21% |
| mlp_legal_mass | 98.48% | 97.58% | 99.34% | 98.90% | 1.09% | -0.67% | 0.45% |
| mlp_legal_top1 | 99.73% | 99.42% | 99.85% | 99.72% | 0.21% | -0.22% | 0.18% |

For legal-move rows, each AR value is its single native-head result; the Linear and MLP rows compare that same AR result against the corresponding frozen JEPA readout.

Interaction is `(Mamba-JEPA − Mamba-AR) − (Transformer-JEPA − Transformer-AR)`. Positive values mean JEPA gains more (or loses less) under Mamba for that metric.

**Comparability gate: PASS.** All four checkpoints report the same training precision.

## Training efficiency

![Training time](training_time_2x2_b8.png)

Times use the chunk-level training logs and exclude checkpoint/Drive synchronization unless it was included inside the recorded chunk time.

Machine-readable results: `/content/drive/MyDrive/Master_Thesis_Artifacts/reports/thesis_eval/b8/factorial_2x2_b8.json`
