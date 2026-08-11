# Master Thesis Code — Final

Clean, reproducible presentation of the Othello world-model thesis while the
original sibling `Master_Thesis_Code` remains the read-only behavioural oracle.

## Structure

```text
Master_Thesis_Code_Final/
├── JEPA_Experimentation/          # complete historical JEPA path
├── Main_Experimental_Setup/       # final 2 × 2 × 3 experiment
├── src/othello_thesis/
│   ├── game_engine/               # shared Othello rules and generator
│   ├── data/
│   ├── models/
│   ├── objectives/
│   ├── training/
│   ├── evaluation/
│   └── probes/
├── tests/equivalence/             # old-vs-final behavioural proofs
└── provenance/                    # source/config/archive hashes
```

The main experiment is Transformer/Mamba × AR/JEPA over 8×8, 12×12, and
16×16. The selected JEPA condition uses the generic `OthelloJEPA` topology:
hard-disjoint one-token targets at every valid boundary, a separately
constructed frozen EMA target, a linear predictor, and position-stratified
InfoNCE at temperature 0.1. The older prefix-grouped “v5” implementation is
historical material, not the final main objective.

## Notebooks

`Main_Experimental_Setup/notebooks/` contains:

- four training notebooks, one per architecture-objective cell;
- `common_evaluation.ipynb` for one selected canonical run;
- `causal_intervention.ipynb` for the independent Li, Nanda, and historical
  adapted causal-intervention comparison;
- `comparative_analysis_v3.ipynb` for within-board comparisons and
  cross-board trajectories of controlled effects.

The common evaluator no longer runs causal interventions. Existing common
evaluation artifacts are left unchanged, while new causal outputs are written
to a separate resumable directory under the selected external run.

Training notebooks default to `BOARD_SIZE = 12` and `RUN_PROFILE = "smoke"`.
Each exposes the actual model factory and constructs the model only once after
seeding. Mamba notebooks pin the official thesis runtime versions.

## Behaviour-preservation contract

The refactor may improve names, boundaries, and readability, but it may not
change model construction order, RNG consumption, data ordering, training
math, checkpoint schemas, evaluation semantics, Markdown, tables, plots, or
output names. See [EQUIVALENCE_MATRIX.md](EQUIVALENCE_MATRIX.md) for the test
criteria and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for the source lock.

Large corpora, checkpoints, and evaluation artifacts remain in
`Master_Thesis_Artifacts`; they are not duplicated into this code repository.

The exact continuation point for any future session is
[REFACTOR_STATUS.md](REFACTOR_STATUS.md).

## Validation

Run the complete local compatibility suite with disposable runtime outside the
synchronized repository:

```powershell
python -m pytest tests/equivalence -q --basetemp <OS_TEMP_PATH>
```

The local suite also locks repeated EMA updates, the source's chunk-boundary
and fraction-checkpoint resume semantics, and the exact notebook-facing and
config-driven CLI artifact trees mirrored to a temporary Drive stand-in. That
mirror verifies synchronization behaviour. The separate Phase 7 location
check confirmed that this repository is backed up at
`Computers/MyLaptop/Master_Thesis_Code_Final`
([Drive folder](https://drive.google.com/drive/folders/1KiEFyzXhupI_xFpAtjphbicw6AlBlIUd)).

Local Mamba equivalence uses the explicit pure-PyTorch test backend. The
official `mamba-ssm==2.3.2.post1` gate passed in an isolated no-write Colab
runtime with the successor source environment lock's Torch `2.11.0+cu128` /
CUDA `12.8` software stack. The synchronized script SHA-256 was
`5dcb249f3153851fd96dd723ed3177200d2bb5f1fd95abddd1a62546259d64e2`.
It constructed the production model and completed the required BF16
forward/backward, gradient clip, and AdamW parameter update with the official
mixers. To reproduce it after mounting this repository and installing the
locked packages, run:

```bash
python tools/smoke_official_mamba.py
```

The script writes no training artifacts and succeeds only after printing
`OFFICIAL_MAMBA_SMOKE_PASS`. It also exercises all 15 official length-one
layers, forces a diagnostic parity drift, and verifies that the fast path stays
enabled through a finite backward pass.
