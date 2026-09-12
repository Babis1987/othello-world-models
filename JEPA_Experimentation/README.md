# JEPA Experimentation

This section preserves the research path that led to the thesis's final JEPA
protocol. It is intentionally separate from `Main_Experimental_Setup`: these
experiments explain how the design was discovered, but they are not additional
cells in the final 2 × 2 architecture/objective comparison.

## Read this boundary first

`archive/source_tree/` is a **byte-exact research archive**, not a second active
Python package. Files there keep their original names, imports, notebook state,
outputs, defaults, and even historical inconsistencies. Rewriting them for
style would erase the evidence needed to reconstruct what was actually run.

The active, thesis-facing implementation lives outside this folder. In
particular, the selected final protocol is the generic `OthelloJEPA` path with
hard-disjoint future targets, all valid absolute positions, in-position
InfoNCE, `K=1`, an EMA target and a linear predictor. Its six board/architecture
configs are deliberately absent here:

- Transformer-JEPA: `jepa_v5_infonce_b{8,12,16}_hd_all_positions.yml`
- Mamba-JEPA: `mamba_jepa_v5_hd_infonce_b{8,12,16}_all_positions.yml`

The older prefix-grouped `jepa_hard_disjoint_infonce.py` is an exploratory v5
branch. Despite the similar name, it is **not** the selected final objective.

## Chronology

| Stage | Main question | What changed | Outcome in the research path |
|---|---|---|---|
| v1 | Can next-state latent prediction learn useful board structure? | VICReg baseline; linear/no-predictor, hard-disjoint/nested, all-position and EMA ablations; Transformer and Mamba variants | Established the baseline and exposed sensitivity to view construction and collapse. |
| v2 | Does a slowly moving target stabilise the representation? | EMA target, Transformer predictor, Smooth L1; nested and hard-disjoint/all-position variants | Improved stability but left a powerful-predictor/shortcut concern. |
| v3 | Was predictor capacity hiding a weak encoder? | Smaller MLP predictor and longer `K=8` horizon | Tested the capacity hypothesis; did not remove the need for a better training signal. |
| v4 | Are nested views enabling shortcut learning? | Disjoint-future views and a multi-position MLP (`K=4`), plus Smooth L1/VICReg variants | Made target separation explicit, while introducing a documented predictor confound. |
| v5 exploratory | Does contrastive discrimination prevent mean-future collapse? | Prefix-positive InfoNCE and prefix-grouped hard-disjoint InfoNCE branches | Motivated the final all-position design, but these archived implementations are not the final protocol. |
| selected v5-style protocol | Can every legal temporal boundary provide a matched contrastive task? | Generic JEPA, hard-disjoint next-token target, per-absolute-position InfoNCE, EMA, linear predictor, `K=1` | Became the canonical JEPA cell in `Main_Experimental_Setup`; excluded from this archive's 43 configs. |
| v6 | Can the next action anchor a multimodal future? | Single-action target, multi-mode predictor and distance-margin loss; all-position follow-up | Useful exploratory evidence, not selected. |
| v7 | Can conditioning the predictor on action improve discrimination? | Action-conditioned Smooth L1, grouped InfoNCE and hybrid CE variants | Explored action information as an explicit conditioning signal. |
| v8 | Can transpositions teach order-invariant board state? | Cross-order positives, same-set hard negatives, pair index and hybrid variants | Tested order awareness directly; retained as a separate research branch. |
| v9 | Can latent state be rolled forward through several actions? | Multi-action rollout with Smooth L1 and InfoNCE variants | Extended the objective beyond one action; not part of the final grid. |
| family InfoNCE | Can transposition families provide the contrastive classes directly? | Offline family artifacts and multi-positive family InfoNCE | Parallel exploratory branch, not a numbered v10 and not part of the final grid. |

This chronology is descriptive, not a claim that every stage was a controlled
single-variable ablation. Several configs explicitly record their confounds.

## Contents

- `archive/source_tree/configs/`: all **43** historical non-canonical JEPA YAML files.
- `archive/source_tree/othello_research/`: 21 variant-specific historical code modules.
- `archive/source_tree/notebooks/`: 26 training, evaluation and diagnostic notebooks.
- `archive/source_tree/reports/`: four small audit/protocol reports.
- `archive/source_tree/results/`: small JSON results/config records for the v1–v5 prefix-mask runs; no checkpoints.
- `archive/source_tree/tests/`: the historical variant tests and regression references.
- `ARCHIVE_INDEX.md`: compact inventory of the retained experimental files.

Large datasets, checkpoints, caches and run directories are excluded. They
remain external research artifacts, as required by the repository rules.

## Exact duplicates retained intentionally

Two duplicate notebook pairs existed in the source working tree. Both paths are
kept because their locations are part of the historical record:

1. `notebooks/diagnostics_jepa.ipynb` and
   `notebooks/Transformer-JEPA/diagnostics_jepa.ipynb`
2. `notebooks/othello_gpt_jepa_evaluation_v2.ipynb` and
   `notebooks/Transformer-JEPA/8x8/othello_gpt_jepa_evaluation_v2.ipynb`

The duplicate paths are retained because both were used during the exploratory
phase and make the original execution history easier to follow.
