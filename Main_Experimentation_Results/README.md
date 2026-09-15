# Main Experimentation Results

Results for the main body of the thesis: the 2 × 2 Architecture × Objective grid
(Transformer/Mamba × AR/JEPA) evaluated at three board sizes, the cross-model
comparison, and the multi-seed replications.

This folder is the counterpart to `JEPA_Experimentation/`. That folder holds the
research path that produced the JEPA protocol; this one holds the results of the
final grid. Nothing here is an ablation or an exploratory variant.

The per-cell, comparison, and seeded report copies were collected from the
external artifact store (`G:\My Drive\Master_Thesis_Artifacts`) by
[`tools/organize_main_experimentation_results.py`](../tools/organize_main_experimentation_results.py).
`provenance/results_manifest.json` records the source path, byte size, and
SHA-256 of each of those copied files. The derived `Chapter_5/` tables and
figures are produced separately and are not entries in that copier manifest.

## Layout

```
Transformer-AR/   8x8/  12x12/  16x16/     per-cell evaluation output
Transformer-JEPA/ 8x8/  12x12/  16x16/
Mamba-AR/         8x8/  12x12/  16x16/
Mamba-JEPA/       8x8/  12x12/  16x16/
Full_Comparison/  all_boards/              retained earlier cross-board v2 snapshot
                  8x8/ 12x12/ 16x16/       per-board 2 × 2 factorial reports
Chapter_5/        chapter5_*.csv|json, figures; derived thesis tables
seeded_results/   <family>/8x8|16x16/      seeds 0-3, one dir per seed
                  aggregate/8x8|16x16/     multi-seed factorial statistics
provenance/       results_manifest.json
```

Inside a cell the original run-relative structure is preserved:

- `thesis_eval/final/` — the headline common evaluation (`unified_eval_v4`):
  `summary__unified_eval_v4.md` carries the legality and board-probe tables,
  checkpoint SHA-256, and stage wall times;
  `proposal_report__thesis_eval_suite_v1.md` is the narrative version with
  phase-stratified tables.
- `causal_intervention/causal_intervention_suite_v2/full/readout_*/` — the
  separate Li/Nanda/adapted comparison, including the corrected
  geometry-matched primary random control. Older v1 results remain where
  present as historical artifacts.
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

The seeded replications use the same configs at seeds 0-3 for 8x8 and 16x16,
all recording `precision: bf16`. The older `runs/Mamba-AR/mamba_ar_8x8`
(fp16) and `runs/run_001` (precision unreported) are not the current
canonical 8x8 AR cells. They do appear in the retained earlier
`Full_Comparison/all_boards/` snapshot described below.

## Current coverage and an older comparison snapshot

**All twelve canonical bf16 seed-42 cells have completed common evaluations.**
The 8x8 `transformer_ar_b8_bf16` and `mamba_ar_8x8_bf16` runs now include
`thesis_eval/final/` summaries and separate causal-suite-v2 results here.
The current `Full_Comparison/8x8/factorial_2x2_b8.md` reports
**Comparability gate: PASS** for four bf16 checkpoints.

**The retained `Full_Comparison/all_boards/` v2 report has not been regenerated
after those evaluations.** Its 8x8 rows still reference `run_001` and
`mamba_ar_8x8`, with mixed or unreported training precision. Do not use its
8x8 effects as the current bf16-controlled comparison. The current per-board
factorial reports and the bf16-backed
[`Chapter_5/chapter5_main_results.csv`](Chapter_5/chapter5_main_results.csv)
are the appropriate published summaries for the canonical cells. The
`comparative_analysis_v3.ipynb` workflow writes a separate v3 cross-board
report; it does not silently update this retained v2 folder.

The `Chapter_5/chapter5_causal_results.csv` aggregate also predates the
corrected primary random control. Until Chapter 5 is regenerated, use the
per-cell causal-suite-v2 summaries for corrected causal comparisons rather
than that older aggregate.

**There are no 12x12 seeded replications.**
`seeded_results/` covers 8x8 and 16x16 only, for all four families. The 12x12
column has a single seed-42 run per cell, so it supports no seed-variance
statement.

## Headline numbers

From the bf16-backed `Chapter_5/chapter5_main_results.csv`, selected legal
readout per cell. Board-probe columns are relative-label macro test accuracy;
values are rounded to two decimal places.

| Cell | Head | Top-1 legal | Legal mass | Linear board | MLP board |
|---|---|---:|---:|---:|---:|
| Transformer-AR 8x8 | Native AR | 99.73% | 98.48% | 96.91% | 96.61% |
| Transformer-JEPA 8x8 | Frozen Linear | 99.42% | 97.76% | 96.72% | 96.41% |
| Mamba-AR 8x8 | Native AR | 99.85% | 99.34% | 97.85% | 97.62% |
| Mamba-JEPA 8x8 | Frozen MLP | 99.72% | 98.90% | 98.04% | 97.82% |
| Transformer-AR 12x12 | Native AR | 99.42% | 93.14% | 89.17% | 88.01% |
| Transformer-JEPA 12x12 | Frozen Linear | 98.52% | 93.30% | 88.48% | 87.40% |
| Mamba-AR 12x12 | Native AR | 99.51% | 95.41% | 91.37% | 90.06% |
| Mamba-JEPA 12x12 | Frozen Linear | 98.99% | 95.47% | 92.85% | 91.85% |
| Transformer-AR 16x16 | Native AR | 98.99% | 88.79% | 82.88% | 80.49% |
| Transformer-JEPA 16x16 | Frozen Linear | 97.54% | 87.93% | 78.37% | 75.99% |
| Mamba-AR 16x16 | Native AR | 99.05% | 90.74% | 83.35% | 81.41% |
| Mamba-JEPA 16x16 | Frozen Linear | 98.41% | 90.42% | 83.77% | 81.26% |

All rows in this table use bf16-trained canonical checkpoints.

## Regenerating

```powershell
python tools/organize_main_experimentation_results.py --dry-run
python tools/organize_main_experimentation_results.py --profile lean --force
```

`--profile full` copies the omitted dumps as well. `--dry-run` prints
the per-cell file counts and sizes and names any cell that has training output
but no evaluation report.
