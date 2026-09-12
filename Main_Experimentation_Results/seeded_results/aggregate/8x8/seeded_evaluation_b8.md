# Multi-seed Common Thesis Evaluation — 8x8

- **Report:** `thesis_seeded_evaluation_v2`
- **Common evaluation protocol:** `unified_eval_v4`
- **Training seeds:** `[0, 1, 2, 3]` (`n=4` per cell)
- **Path-independent split identity:** `7b95882e8f84aa12e078ef4723e59bb64fe29debf4a07f85d4102d991e875833`
- **Data manifest:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Split audit:** corpus manifest plus ordered shard filenames are identical across all runs; raw absolute-path hashes may differ because every seed uses an isolated evaluation view.
- **Generated UTC:** `2026-08-09T17:07:14.037497+00:00`
- **Design:** balanced, matched 2x2 Architecture x Objective; seed is the block.
- **Uncertainty unit:** independent training runs. Per-game bootstrap intervals from the Common Evaluator are not used as substitutes for seed variance.

## Executive view

- **Legal top-1 / best available head:** highest mean is **Mamba-AR** at **99.84%** (SD 0.01%; seed-level 95% t CI 99.81%–99.86%).
- **Legal probability mass / best available head:** highest mean is **Mamba-AR** at **99.38%** (SD 0.06%; seed-level 95% t CI 99.29%–99.48%).
- **Mean board-state macro accuracy:** highest mean is **Mamba-JEPA** at **89.87%** (SD 0.13%; seed-level 95% t CI 89.66%–90.08%).

Primary factorial effects surviving Holm correction:
- Legal top-1 / best available head — **Mamba - Transformer (main effect)**: +0.22 pp, 95% CI [+0.21 pp, +0.22 pp], Holm q=<0.001.
- Legal top-1 / best available head — **JEPA - AR (main effect)**: -0.20 pp, 95% CI [-0.21 pp, -0.19 pp], Holm q=<0.001.
- Legal top-1 / best available head — **Architecture x Objective interaction**: +0.18 pp, 95% CI [+0.14 pp, +0.22 pp], Holm q=0.004.
- Legal probability mass / best available head — **Mamba - Transformer (main effect)**: +1.09 pp, 95% CI [+0.93 pp, +1.25 pp], Holm q=0.001.
- Legal probability mass / best available head — **JEPA - AR (main effect)**: -0.53 pp, 95% CI [-0.60 pp, -0.46 pp], Holm q=0.001.
- Legal probability mass / best available head — **Architecture x Objective interaction**: +0.28 pp, 95% CI [+0.19 pp, +0.38 pp], Holm q=0.010.
- Mean board-state macro accuracy — **Architecture x Objective interaction**: +1.38 pp, 95% CI [+0.87 pp, +1.88 pp], Holm q=0.010.

## What is being tested

The three factorial contrasts are computed independently inside every seed and then tested against zero:

- Architecture main effect: average `(Mamba - Transformer)` across AR and JEPA.
- Objective main effect: average `(JEPA - AR)` across Transformer and Mamba.
- Interaction: `(Mamba-JEPA - Mamba-AR) - (Transformer-JEPA - Transformer-AR)`.

The paired t-test is the parametric inferential test; 95% t intervals and Hedges' `gz` report magnitude and uncertainty. The exact sign-flip p-value is a distribution-light robustness check. With `n` seeds, its smallest possible two-sided value is `2 / 2^n`; therefore it cannot cross 0.05 with four or five seeds. Holm correction is applied across the 3 metrics x 3 effects primary family and separately across the secondary family. Pairwise tests are exploratory and Holm-corrected within each metric.

## Per-seed results

### Legal top-1 / best available head

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 99.71% | 99.72% | 99.70% | 99.71% | 99.71% ± 0.01% | [99.69%, 99.73%] |
| Transformer-JEPA | 99.40% | 99.43% | 99.42% | 99.43% | 99.42% ± 0.02% | [99.40%, 99.44%] |
| Mamba-AR | 99.82% | 99.85% | 99.83% | 99.84% | 99.84% ± 0.01% | [99.81%, 99.86%] |
| Mamba-JEPA | 99.72% | 99.72% | 99.73% | 99.73% | 99.73% ± 0.00% | [99.72%, 99.73%] |

### Legal probability mass / best available head

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 98.42% | 98.35% | 98.47% | 98.50% | 98.44% ± 0.07% | [98.33%, 98.54%] |
| Transformer-JEPA | 97.76% | 97.66% | 97.76% | 97.87% | 97.76% ± 0.09% | [97.62%, 97.90%] |
| Mamba-AR | 99.30% | 99.43% | 99.42% | 99.37% | 99.38% ± 0.06% | [99.29%, 99.48%] |
| Mamba-JEPA | 99.01% | 99.03% | 98.97% | 98.97% | 98.99% ± 0.03% | [98.95%, 99.04%] |

### Mean board-state macro accuracy

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 89.03% | 89.21% | 89.27% | 89.11% | 89.15% ± 0.11% | [88.98%, 89.33%] |
| Transformer-JEPA | 89.38% | 89.00% | 89.15% | 88.97% | 89.12% ± 0.19% | [88.82%, 89.43%] |
| Mamba-AR | 87.78% | 88.55% | 89.05% | 88.72% | 88.52% ± 0.54% | [87.67%, 89.38%] |
| Mamba-JEPA | 89.83% | 89.75% | 89.86% | 90.06% | 89.87% ± 0.13% | [89.66%, 90.08%] |

### Board Linear / absolute

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 68.94% | 68.90% | 68.91% | 68.89% | 68.91% ± 0.02% | [68.88%, 68.94%] |
| Transformer-JEPA | 69.01% | 69.03% | 68.97% | 69.08% | 69.02% ± 0.05% | [68.95%, 69.10%] |
| Mamba-AR | 68.68% | 68.74% | 68.71% | 68.81% | 68.74% ± 0.05% | [68.65%, 68.82%] |
| Mamba-JEPA | 68.93% | 68.96% | 69.09% | 69.06% | 69.01% ± 0.08% | [68.88%, 69.13%] |

### Board Linear / relative

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 96.88% | 96.94% | 97.05% | 96.98% | 96.96% ± 0.07% | [96.85%, 97.08%] |
| Transformer-JEPA | 96.89% | 96.38% | 96.74% | 96.34% | 96.59% ± 0.27% | [96.16%, 97.02%] |
| Mamba-AR | 97.96% | 98.02% | 98.05% | 97.71% | 97.93% ± 0.16% | [97.69%, 98.18%] |
| Mamba-JEPA | 98.21% | 98.22% | 98.19% | 98.21% | 98.21% ± 0.02% | [98.18%, 98.23%] |

### Board MLP / absolute

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 93.74% | 94.43% | 94.36% | 93.85% | 94.10% ± 0.35% | [93.54%, 94.66%] |
| Transformer-JEPA | 95.08% | 94.52% | 94.44% | 94.40% | 94.61% ± 0.32% | [94.10%, 95.11%] |
| Mamba-AR | 86.71% | 89.64% | 91.54% | 90.79% | 89.67% ± 2.12% | [86.30%, 93.05%] |
| Mamba-JEPA | 94.12% | 93.75% | 94.16% | 94.94% | 94.24% ± 0.50% | [93.45%, 95.04%] |

### Board MLP / relative

| System | seed 0 | seed 1 | seed 2 | seed 3 | Mean ± SD | Seed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Transformer-AR | 96.55% | 96.58% | 96.77% | 96.70% | 96.65% ± 0.10% | [96.49%, 96.81%] |
| Transformer-JEPA | 96.56% | 96.07% | 96.44% | 96.06% | 96.28% ± 0.26% | [95.87%, 96.69%] |
| Mamba-AR | 97.77% | 97.81% | 97.88% | 97.56% | 97.76% ± 0.14% | [97.54%, 97.98%] |
| Mamba-JEPA | 98.05% | 98.04% | 97.99% | 98.03% | 98.03% ± 0.03% | [97.98%, 98.07%] |

## Matched 2x2 factorial inference

| Metric | Effect | Mean | 95% CI | Hedges gz | p (paired t) | Holm q | Exact sign-flip p | Direction wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Legal top-1 / best available head | Mamba - Transformer (main effect) | +0.22 pp | [+0.21 pp, +0.22 pp] | +38.06 | <0.001 | <0.001 **✓** | 0.125 | 4/0/0 |
| Legal top-1 / best available head | JEPA - AR (main effect) | -0.20 pp | [-0.21 pp, -0.19 pp] | -17.42 | <0.001 | <0.001 **✓** | 0.125 | 0/0/4 |
| Legal top-1 / best available head | Architecture x Objective interaction | +0.18 pp | [+0.14 pp, +0.22 pp] | +5.20 | <0.001 | 0.004 **✓** | 0.125 | 4/0/0 |
| Legal probability mass / best available head | Mamba - Transformer (main effect) | +1.09 pp | [+0.93 pp, +1.25 pp] | +7.91 | <0.001 | 0.001 **✓** | 0.125 | 4/0/0 |
| Legal probability mass / best available head | JEPA - AR (main effect) | -0.53 pp | [-0.60 pp, -0.46 pp] | -8.54 | <0.001 | 0.001 **✓** | 0.125 | 0/0/4 |
| Legal probability mass / best available head | Architecture x Objective interaction | +0.28 pp | [+0.19 pp, +0.38 pp] | +3.39 | 0.003 | 0.010 **✓** | 0.125 | 4/0/0 |
| Mean board-state macro accuracy | Mamba - Transformer (main effect) | +0.06 pp | [-0.47 pp, +0.59 pp] | +0.13 | 0.747 | 0.747 | 0.875 | 3/0/1 |
| Mean board-state macro accuracy | JEPA - AR (main effect) | +0.66 pp | [+0.06 pp, +1.26 pp] | +1.27 | 0.040 | 0.080 | 0.125 | 4/0/0 |
| Mean board-state macro accuracy | Architecture x Objective interaction | +1.38 pp | [+0.87 pp, +1.88 pp] | +3.16 | 0.003 | 0.010 **✓** | 0.125 | 4/0/0 |
| Board Linear / absolute | Mamba - Transformer (main effect) | -0.09 pp | [-0.19 pp, +0.00 pp] | -1.10 | 0.056 | 0.247 | 0.125 | 0/0/4 |
| Board Linear / absolute | JEPA - AR (main effect) | +0.19 pp | [+0.14 pp, +0.24 pp] | +4.41 | 0.001 | 0.012 **✓** | 0.125 | 4/0/0 |
| Board Linear / absolute | Architecture x Objective interaction | +0.16 pp | [-0.02 pp, +0.34 pp] | +1.02 | 0.068 | 0.247 | 0.125 | 4/0/0 |
| Board Linear / relative | Mamba - Transformer (main effect) | +1.30 pp | [+1.11 pp, +1.48 pp] | +7.95 | <0.001 | 0.002 **✓** | 0.125 | 4/0/0 |
| Board Linear / relative | JEPA - AR (main effect) | -0.05 pp | [-0.25 pp, +0.15 pp] | -0.29 | 0.490 | 0.980 | 0.500 | 1/0/3 |
| Board Linear / relative | Architecture x Objective interaction | +0.65 pp | [+0.02 pp, +1.28 pp] | +1.19 | 0.047 | 0.247 | 0.125 | 4/0/0 |
| Board MLP / absolute | Mamba - Transformer (main effect) | -2.40 pp | [-4.38 pp, -0.41 pp] | -1.39 | 0.031 | 0.235 | 0.125 | 0/0/4 |
| Board MLP / absolute | JEPA - AR (main effect) | +2.54 pp | [+0.48 pp, +4.60 pp] | +1.43 | 0.029 | 0.235 | 0.125 | 4/0/0 |
| Board MLP / absolute | Architecture x Objective interaction | +4.06 pp | [+1.71 pp, +6.42 pp] | +2.00 | 0.012 | 0.107 | 0.125 | 4/0/0 |
| Board MLP / relative | Mamba - Transformer (main effect) | +1.43 pp | [+1.23 pp, +1.62 pp] | +8.44 | <0.001 | 0.002 **✓** | 0.125 | 4/0/0 |
| Board MLP / relative | JEPA - AR (main effect) | -0.05 pp | [-0.26 pp, +0.16 pp] | -0.28 | 0.502 | 0.980 | 0.625 | 1/0/3 |
| Board MLP / relative | Architecture x Objective interaction | +0.64 pp | [+0.05 pp, +1.23 pp] | +1.25 | 0.041 | 0.247 | 0.125 | 4/0/0 |

## Planned paired comparisons

Positive differences favor the first named system/group. Pairwise Holm correction is performed separately for each metric.

| Metric | Contrast | Mean | 95% CI | Hedges gz | p (paired t) | Holm q | Exact p |
|---|---|---:|---:|---:|---:|---:|---:|
| Legal top-1 / best available head | Mamba - Transformer | AR | +0.13 pp | [+0.11 pp, +0.15 pp] | +7.30 | <0.001 | <0.001 **✓** | 0.125 |
| Legal top-1 / best available head | Mamba - Transformer | JEPA | +0.31 pp | [+0.28 pp, +0.33 pp] | +16.09 | <0.001 | <0.001 **✓** | 0.125 |
| Legal top-1 / best available head | JEPA - AR | Transformer | -0.29 pp | [-0.31 pp, -0.26 pp] | -13.34 | <0.001 | <0.001 **✓** | 0.125 |
| Legal top-1 / best available head | JEPA - AR | Mamba | -0.11 pp | [-0.13 pp, -0.09 pp] | -5.60 | <0.001 | 0.001 **✓** | 0.125 |
| Legal top-1 / best available head | Mamba-JEPA - Transformer-AR | +0.02 pp | [-0.00 pp, +0.04 pp] | +0.96 | 0.077 | 0.077 | 0.125 |
| Legal top-1 / best available head | Mamba-AR - Transformer-JEPA | +0.42 pp | [+0.41 pp, +0.42 pp] | +56.64 | <0.001 | <0.001 **✓** | 0.125 |
| Legal probability mass / best available head | Mamba - Transformer | AR | +0.95 pp | [+0.79 pp, +1.10 pp] | +7.11 | <0.001 | <0.001 **✓** | 0.125 |
| Legal probability mass / best available head | Mamba - Transformer | JEPA | +1.23 pp | [+1.05 pp, +1.41 pp] | +7.99 | <0.001 | <0.001 **✓** | 0.125 |
| Legal probability mass / best available head | JEPA - AR | Transformer | -0.67 pp | [-0.73 pp, -0.61 pp] | -12.60 | <0.001 | <0.001 **✓** | 0.125 |
| Legal probability mass / best available head | JEPA - AR | Mamba | -0.39 pp | [-0.49 pp, -0.28 pp] | -4.26 | 0.001 | 0.002 **✓** | 0.125 |
| Legal probability mass / best available head | Mamba-JEPA - Transformer-AR | +0.56 pp | [+0.41 pp, +0.71 pp] | +4.38 | 0.001 | 0.002 **✓** | 0.125 |
| Legal probability mass / best available head | Mamba-AR - Transformer-JEPA | +1.62 pp | [+1.42 pp, +1.82 pp] | +9.45 | <0.001 | <0.001 **✓** | 0.125 |
| Mean board-state macro accuracy | Mamba - Transformer | AR | -0.63 pp | [-1.34 pp, +0.08 pp] | -1.03 | 0.067 | 0.200 | 0.125 |
| Mean board-state macro accuracy | Mamba - Transformer | JEPA | +0.75 pp | [+0.32 pp, +1.17 pp] | +2.04 | 0.011 | 0.056 | 0.125 |
| Mean board-state macro accuracy | JEPA - AR | Transformer | -0.03 pp | [-0.45 pp, +0.39 pp] | -0.08 | 0.831 | 0.831 | 1.000 |
| Mean board-state macro accuracy | JEPA - AR | Mamba | +1.35 pp | [+0.53 pp, +2.17 pp] | +1.90 | 0.014 | 0.056 | 0.125 |
| Mean board-state macro accuracy | Mamba-JEPA - Transformer-AR | +0.72 pp | [+0.41 pp, +1.03 pp] | +2.66 | 0.005 | 0.032 **✓** | 0.125 |
| Mean board-state macro accuracy | Mamba-AR - Transformer-JEPA | -0.60 pp | [-1.69 pp, +0.49 pp] | -0.64 | 0.177 | 0.355 | 0.125 |
| Board Linear / absolute | Mamba - Transformer | AR | -0.17 pp | [-0.28 pp, -0.06 pp] | -1.82 | 0.015 | 0.062 | 0.125 |
| Board Linear / absolute | Mamba - Transformer | JEPA | -0.01 pp | [-0.16 pp, +0.14 pp] | -0.09 | 0.823 | 0.823 | 0.875 |
| Board Linear / absolute | JEPA - AR | Transformer | +0.11 pp | [+0.02 pp, +0.20 pp] | +1.43 | 0.029 | 0.088 | 0.125 |
| Board Linear / absolute | JEPA - AR | Mamba | +0.27 pp | [+0.16 pp, +0.39 pp] | +2.69 | 0.005 | 0.026 **✓** | 0.125 |
| Board Linear / absolute | Mamba-JEPA - Transformer-AR | +0.10 pp | [-0.05 pp, +0.25 pp] | +0.79 | 0.118 | 0.236 | 0.250 |
| Board Linear / absolute | Mamba-AR - Transformer-JEPA | -0.28 pp | [-0.33 pp, -0.23 pp] | -6.57 | <0.001 | 0.002 **✓** | 0.125 |
| Board Linear / relative | Mamba - Transformer | AR | +0.97 pp | [+0.71 pp, +1.24 pp] | +4.23 | 0.001 | 0.007 **✓** | 0.125 |
| Board Linear / relative | Mamba - Transformer | JEPA | +1.62 pp | [+1.17 pp, +2.07 pp] | +4.20 | 0.001 | 0.007 **✓** | 0.125 |
| Board Linear / relative | JEPA - AR | Transformer | -0.37 pp | [-0.83 pp, +0.09 pp] | -0.94 | 0.082 | 0.086 | 0.250 |
| Board Linear / relative | JEPA - AR | Mamba | +0.27 pp | [+0.02 pp, +0.53 pp] | +1.23 | 0.043 | 0.086 | 0.125 |
| Board Linear / relative | Mamba-JEPA - Transformer-AR | +1.25 pp | [+1.11 pp, +1.38 pp] | +10.91 | <0.001 | <0.001 **✓** | 0.125 |
| Board Linear / relative | Mamba-AR - Transformer-JEPA | +1.35 pp | [+0.98 pp, +1.71 pp] | +4.25 | 0.001 | 0.007 **✓** | 0.125 |
| Board MLP / absolute | Mamba - Transformer | AR | -4.43 pp | [-7.52 pp, -1.33 pp] | -1.66 | 0.020 | 0.119 | 0.125 |
| Board MLP / absolute | Mamba - Transformer | JEPA | -0.36 pp | [-1.43 pp, +0.70 pp] | -0.40 | 0.355 | 0.709 | 0.375 |
| Board MLP / absolute | JEPA - AR | Transformer | +0.51 pp | [-0.43 pp, +1.45 pp] | +0.62 | 0.184 | 0.553 | 0.125 |
| Board MLP / absolute | JEPA - AR | Mamba | +4.57 pp | [+1.35 pp, +7.79 pp] | +1.64 | 0.020 | 0.119 | 0.125 |
| Board MLP / absolute | Mamba-JEPA - Transformer-AR | +0.15 pp | [-1.07 pp, +1.36 pp] | +0.14 | 0.729 | 0.729 | 0.750 |
| Board MLP / absolute | Mamba-AR - Transformer-JEPA | -4.94 pp | [-8.80 pp, -1.07 pp] | -1.48 | 0.027 | 0.119 | 0.125 |
| Board MLP / relative | Mamba - Transformer | AR | +1.11 pp | [+0.84 pp, +1.38 pp] | +4.71 | <0.001 | 0.004 **✓** | 0.125 |
| Board MLP / relative | Mamba - Transformer | JEPA | +1.75 pp | [+1.33 pp, +2.17 pp] | +4.80 | <0.001 | 0.004 **✓** | 0.125 |
| Board MLP / relative | JEPA - AR | Transformer | -0.37 pp | [-0.82 pp, +0.08 pp] | -0.95 | 0.079 | 0.079 | 0.250 |
| Board MLP / relative | JEPA - AR | Mamba | +0.27 pp | [+0.03 pp, +0.51 pp] | +1.29 | 0.038 | 0.076 | 0.125 |
| Board MLP / relative | Mamba-JEPA - Transformer-AR | +1.38 pp | [+1.17 pp, +1.58 pp] | +7.77 | <0.001 | 0.001 **✓** | 0.125 |
| Board MLP / relative | Mamba-AR - Transformer-JEPA | +1.48 pp | [+1.13 pp, +1.82 pp] | +4.92 | <0.001 | 0.004 **✓** | 0.125 |

## Visual diagnostics

![Seed distributions](seed_distributions_b8.png)

![Factorial effect forest](factorial_forest_b8.png)

![Probability of superiority](seed_win_heatmap_b8.png)

## Interpretation boundaries

- AR legality uses its native co-trained next-move head; JEPA legality uses the validation-selected frozen Linear/MLP readout. Those are system-level comparisons, not encoder-matched readout comparisons.
- Board-state probes compare frozen encoder representations under common probe budgets and are the cleaner encoder-state comparison.
- All runs share one fixed test split. More test games reduce test-set noise but do not increase the number of independent training replicates.
- A non-significant result does not establish equivalence. Equivalence requires a predeclared smallest effect size of interest and a dedicated equivalence test, preferably with more seeds.
- Statistical significance is reported together with effect size and interval; it is not used as a substitute for scientific importance.

## Artifact index

- Machine-readable report: `/content/drive/MyDrive/Master_Thesis_Artifacts/reports/thesis_eval/multiseed/b8/seeded_evaluation_b8.json`
- Raw per-seed metrics: `/content/drive/MyDrive/Master_Thesis_Artifacts/reports/thesis_eval/multiseed/b8/raw_per_seed_b8.csv`
- Cell summaries: `/content/drive/MyDrive/Master_Thesis_Artifacts/reports/thesis_eval/multiseed/b8/cell_summaries_b8.csv`
- Factorial effects: `/content/drive/MyDrive/Master_Thesis_Artifacts/reports/thesis_eval/multiseed/b8/factorial_effects_b8.csv`
- Pairwise effects: `/content/drive/MyDrive/Master_Thesis_Artifacts/reports/thesis_eval/multiseed/b8/pairwise_effects_b8.csv`
- Replication protocol: `/content/drive/Othercomputers/MyLaptop/Master_Thesis_Code/configs/seeded_replication_protocol.yml`

