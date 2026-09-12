# Main Experimentation Results

Results for the main body of the thesis: the 2 × 2 Architecture × Objective grid
(Transformer/Mamba × AR/JEPA) evaluated at three board sizes, the cross-model
comparison, and the multi-seed replications.

This folder is the counterpart to `JEPA_Experimentation/`. That folder holds the
research path that produced the JEPA protocol; this one holds the results of the
final grid. Nothing here is an ablation or an exploratory variant.

Everything was copied from the external artifact store
(`G:\My Drive\Master_Thesis_Artifacts`) by
[`tools/organize_main_experimentation_results.py`](../tools/organize_main_experimentation_results.py).
`provenance/results_manifest.json` records the source path, byte size and
SHA-256 of every file.

## Layout

```
Transformer-AR/   8x8/  12x12/  16x16/     per-cell evaluation output
Transformer-JEPA/ 8x8/  12x12/  16x16/
Mamba-AR/         8x8/  12x12/  16x16/
Mamba-JEPA/       8x8/  12x12/  16x16/
Full_Comparison/  all_boards/              4 models × 3 boards, unified_eval_v4
                  8x8/ 12x12/ 16x16/       per-board 2 × 2 factorial report
seeded_results/   <family>/8x8|16x16/      seeds 0-3, one dir per seed
                  aggregate/8x8|16x16/     multi-seed factorial statistics
provenance/       results_manifest.json
```

Inside a cell the original run-relative structure is preserved:

- `thesis_eval/final/` — the headline evaluation (`unified_eval_v4`):
  `summary__unified_eval_v4.md` carries the frozen-head legality table, the
  full per-layer board-probe tables, the causal-intervention result, checkpoint
  SHA-256 and stage wall times; `proposal_report__thesis_eval_suite_v1.md` is
  the narrative version with phase-stratified tables.
- `causal_intervention/causal_intervention_suite_v1/full/` — the Nanda-style
  intervention suite.
- `unified_eval/final/` — superseded `unified_eval_v2` output, where a run has it.
- `eval_plots/` — the older per-run diagnostic plots, where a run has them.
- `train_config.json`, `metrics.csv` — training configuration and chunk log.

## Which run backs which cell

All twelve are seed 42, `final.pt`, trained bf16.

| Cell | Drive run | Training precision |
|---|---|---|
| Transformer-AR 8x8 | `runs/Transformer-AR/transformer_ar_b8_bf16` | bf16 |
| Transformer-AR 12x12 | `runs/Transformer-AR/transformer_ar_b12` | bf16 |
| Transformer-AR 16x16 | `runs/Transformer-AR/transformer_ar_b16` | bf16 |
| Transformer-JEPA 8x8 | `runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16` | bf16 |
| Transformer-JEPA 12x12 | `runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12` | bf16 |
| Transformer-JEPA 16x16 | `runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16` | bf16 |
| Mamba-AR 8x8 | `runs/Mamba-AR/mamba_ar_8x8_bf16` | bf16 |
| Mamba-AR 12x12 | `runs/Mamba-AR/mamba_ar_b12` | bf16 |
| Mamba-AR 16x16 | `runs/Mamba-AR/mamba_ar_b16` | bf16 |
| Mamba-JEPA 8x8 | `runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8` | bf16 |
| Mamba-JEPA 12x12 | `runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12` | bf16 |
| Mamba-JEPA 16x16 | `runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16` | bf16 |

The seeded replications use the same twelve configs at seeds 0-3, all recording
`precision: bf16`. The older `runs/Mamba-AR/mamba_ar_8x8` (fp16) and
`runs/run_001` (precision unreported, so fp32) are deliberately **not** mapped
to any cell — see the gap below.

## Two gaps

**1. The 8x8 AR cells are trained in bf16 but never evaluated.**
`transformer_ar_b8_bf16` and `mamba_ar_8x8_bf16` hold checkpoints, `metrics.csv`
and `train_config.json`, and nothing else — no `thesis_eval`, no
`causal_intervention`. Their folders here contain only those two files.

The consequence is that the published 8x8 comparison does not use them. The
`Full_Comparison/all_boards/metrics.csv` rows for 8x8 come from `run_001`
(Transformer-AR, precision unreported) and `mamba_ar_8x8` (fp16), and
`Full_Comparison/8x8/factorial_2x2_b8.md` flags this itself:

> **Comparability gate: WARNING.** Training precision is mixed or unreported.

12x12 and 16x16 both report **PASS** — all four checkpoints bf16.

No retraining is needed — the bf16 checkpoints are already on Drive, and
`Main_Experimental_Setup/configs/thesis_evaluation_registry.yml` already lists
them ahead of the legacy runs for both cells. Closing the gap means running, on
a GPU:

```bash
python scripts/run_thesis_evaluation.py --architecture transformer --objective ar --board-size 8
python scripts/run_thesis_evaluation.py --architecture mamba --objective ar --board-size 8
python scripts/run_causal_intervention.py --architecture transformer --objective ar --board-size 8
python scripts/run_causal_intervention.py --architecture mamba --objective ar --board-size 8
```

then regenerating the 8x8 factorial and the all-boards comparison from
`Main_Experimental_Setup/notebooks/comparative_analysis_v3.ipynb`, and re-running
the copier here.

Note the seeded 8x8 replications (seeds 0-3) of these same configs *are*
evaluated and *are* bf16, so `seeded_results/*/8x8/` and
`seeded_results/aggregate/8x8/` carry no precision caveat.

**2. There are no 12x12 seeded replications.**
`seeded_results/` covers 8x8 and 16x16 only, for all four families. The 12x12
column has a single seed-42 run per cell, so it supports no seed-variance
statement.

## Headline numbers

From `Full_Comparison/all_boards/metrics.csv`, best available head per cell.
Board-probe columns are relative-label test accuracy.

| Cell | Head | Top-1 legal | Legal mass | Linear board | MLP board |
|---|---|---:|---:|---:|---:|
| Transformer-AR 8x8 | Native AR | 99.72% | 98.43% | 96.83% | 96.52% |
| Transformer-JEPA 8x8 | Frozen Linear | 99.42% | 97.76% | 96.72% | 96.41% |
| Mamba-AR 8x8 | Native AR | 99.84% | 99.34% | 97.94% | 97.74% |
| Mamba-JEPA 8x8 | Frozen MLP | 99.72% | 98.90% | 98.04% | 97.82% |
| Transformer-AR 12x12 | Native AR | 99.42% | 93.13% | 89.17% | 88.01% |
| Transformer-JEPA 12x12 | Frozen Linear | 98.52% | 93.30% | 88.48% | 87.40% |
| Mamba-AR 12x12 | Native AR | 99.51% | 95.41% | 91.37% | 90.06% |
| Mamba-JEPA 12x12 | Frozen Linear | 98.99% | 95.47% | 92.85% | 91.85% |
| Transformer-AR 16x16 | Native AR | 98.99% | 88.78% | 82.88% | 80.49% |
| Transformer-JEPA 16x16 | Frozen Linear | 97.54% | 87.93% | 78.37% | 75.99% |
| Mamba-AR 16x16 | Native AR | 99.05% | 90.73% | 83.35% | 81.41% |
| Mamba-JEPA 16x16 | Frozen Linear | 98.41% | 90.42% | 83.77% | 81.26% |

The 8x8 rows carry the precision caveat above.

## What was copied, and what stayed on Drive

Copied: `.md`, `.png`, `.csv`, `.tex`, and the small `.json` files
(`train_config.json`, `factorial_2x2_*.json`, `seeded_evaluation_*.json`,
`replication_manifest.json`, `results__causal_intervention_suite_v1.json`).
425 files, 52 MiB.

Left on Drive, because copying them costs 3.6 GB and adds no readable number:

- `results__unified_eval_v*.json` — the per-position dump behind each summary.
  The sibling `summary__*.md` already contains every metric it reports.
- `position_manifest__unified_eval_v*.json` — the sampled evaluation position
  set, an evaluation *input* rather than a result. The 63 `unified_eval_v4`
  copies alone total 1.77 GB but hash to only four distinct contents (one per
  board size plus one smoke variant, 86 MiB together), so 1.7 GB of that is
  exact duplication.
- `case_manifest__causal_intervention_suite_v1.json` — the intervention case
  list, likewise an input.
- `.pt` checkpoints and probe weights, which the repository excludes by rule.

Each omitted file's Drive path is reachable from the "Artifacts and timing"
section at the end of every `summary__*.md`.

## Regenerating

```powershell
python tools/organize_main_experimentation_results.py --dry-run
python tools/organize_main_experimentation_results.py --profile lean --force
```

`--profile full` copies the omitted dumps as well (3.6 GB). `--dry-run` prints
the per-cell file counts and sizes and names any cell that has training output
but no evaluation report.
