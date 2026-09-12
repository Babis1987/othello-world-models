# Othello World Models

Research code, experiment definitions, and selected results for an MSc thesis
on the internal representations learned by sequence models from Othello move
histories.

The central question is deliberately simple: **if a model sees only the moves
of a game, can it recover and use the board state that produced them?** The
repository studies that question by comparing two sequence architectures and
two training objectives across three board sizes.

## Experimental design

| | Autoregressive (AR) | JEPA |
|---|---|---|
| **Transformer** | Transformer-AR | Transformer-JEPA |
| **Mamba** | Mamba-AR | Mamba-JEPA |

Each of these four systems is trained on `8x8`, `12x12`, and `16x16` Othello,
forming the thesis's main `2 x 2 x 3` comparison. The canonical runs use the
same data budget, approximately matched encoder parameter counts, BF16
training, and seed `42`. Independent runs with seeds `0`, `1`, `2`, `3`, and
`42` are available for the `8x8` and `16x16` comparisons.

The final JEPA setup uses:

- a context encoder that receives the move prefix;
- a frozen target encoder updated by exponential moving average (EMA);
- a hard-disjoint target containing only the next move;
- a linear predictor;
- position-stratified InfoNCE at every valid sequence boundary.

Earlier JEPA variants are preserved separately as the development history and
are not mixed with the final factorial experiment.

## What is evaluated

The evaluation is organised in three complementary levels:

1. **Functional behaviour** — whether the model assigns probability to legal
   next moves, measured through top-k legality and total legal probability
   mass.
2. **Board-state decodability** — whether linear and MLP probes can recover the
   board from frozen internal representations.
3. **Causal use** — whether a targeted change to the representation moves the
   prediction toward the legal moves of a counterfactual board.

This distinction matters: a legal prediction, a decodable board state, and a
causally effective representation provide different kinds of evidence. None
of them alone establishes a complete or human-interpretable world model.

## Main findings

- All twelve canonical conditions achieve very high top-1 legal-move accuracy.
- Increasing board size affects the full probability distribution and
  board-state decoding more strongly than it affects the first prediction.
- Mamba-AR gives the highest top-1 legality at all three board sizes.
- Mamba-JEPA gives the highest linear decoding accuracy for relative board
  state at all three board sizes.
- The targeted adapted intervention outperforms its geometry-matched random
  control in all twelve conditions, supporting a causal role for at least part
  of the decodable information.

The Othello corpora contain randomly generated legal games rather than expert
strategy. The results therefore concern rule-consistent state tracking, not
playing strength or optimal decision-making.

## Repository structure

```text
othello-world-models/
|-- Main_Experimental_Setup/
|   |-- configs/                  # locked training and evaluation definitions
|   |-- notebooks/                # canonical training and evaluation workflow
|   `-- reference_contracts/      # resolved machine-checkable protocol
|-- JEPA_Experimentation/         # JEPA development path and comparisons
|-- Main_Experimentation_Results/ # selected summaries, tables, and figures
|-- src/othello_thesis/
|   |-- game_engine/              # Othello rules and deterministic generation
|   |-- data/                     # corpus loading and validation
|   |-- models/                   # Transformer, Mamba, and JEPA components
|   |-- objectives/               # canonical learning objectives
|   |-- training/                 # shared training implementations
|   |-- evaluation/               # unified and causal evaluation
|   `-- probes/                   # board-state probes
|-- scripts/                      # command-line entry points
|-- tests/
|   |-- unit/                     # standalone smoke tests
|   `-- equivalence/              # detailed behavioural contracts
`-- docs/                         # thesis material, figures, and codebase atlas
```

More detail is available in:

- [`Main_Experimental_Setup/README.md`](Main_Experimental_Setup/README.md)
- [`JEPA_Experimentation/README.md`](JEPA_Experimentation/README.md)
- [`Main_Experimentation_Results/README.md`](Main_Experimentation_Results/README.md)
- the offline [`codebase atlas`](docs/codebase_atlas/index.html)

## Installation

Python `3.10` or newer is required. For the Transformer code, evaluation, and
notebooks:

```bash
git clone https://github.com/Babis1987/othello-world-models.git
cd othello-world-models
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
```

The production Mamba experiments require a Linux CUDA environment and the
versions used by the thesis:

```bash
python -m pip install packaging ninja
python -m pip install "causal-conv1d==1.6.2.post1" \
  "mamba-ssm==2.3.2.post1" --no-build-isolation
python -m pip install -e ".[notebooks,mamba]"
```

Google Colab was used for the reported GPU runs. Large corpora, checkpoints,
and complete run directories are intentionally not stored in Git. The
notebooks use a separate artifact root, normally
`/content/drive/MyDrive/Master_Thesis_Artifacts`.

## Reproducing the workflow

### 1. Generate or validate a corpus

```bash
python scripts/generate_corpus.py \
  --board-size 8 \
  --num-games 1000 \
  --output-dir data/demo_8x8

python scripts/audit_corpus.py --help
```

The reported experiments use pre-generated, manifest-checked corpora. The
small command above is only a local pipeline example.

### 2. Train a canonical model

The recommended entry points are the four notebooks in
[`Main_Experimental_Setup/notebooks`](Main_Experimental_Setup/notebooks):

| Notebook | Fixed condition |
|---|---|
| `train_transformer_ar.ipynb` | Transformer + AR |
| `train_transformer_jepa.ipynb` | Transformer + final JEPA |
| `train_mamba_ar.ipynb` | Mamba + AR |
| `train_mamba_jepa.ipynb` | Mamba + final JEPA |

Each notebook exposes `BOARD_SIZE`, `SEED`, and a smoke/full run profile near
the beginning. Set its visible project root to the directory containing this
checkout and keep training artifacts outside the repository.

Equivalent command-line frontends are available under `scripts/`:

```bash
python scripts/train_transformer_ar.py --help
python scripts/train_transformer_jepa.py --help
python scripts/train_mamba_ar.py --help
python scripts/train_mamba_jepa.py --help
```

### 3. Run the common evaluation

[`common_evaluation.ipynb`](Main_Experimental_Setup/notebooks/common_evaluation.ipynb)
runs frozen next-move readouts, legality metrics, board-state probes, and the
shared summary for one selected condition. The same pipeline is available from
the command line:

```bash
python scripts/run_thesis_evaluation.py \
  --architecture transformer \
  --objective jepa \
  --board-size 8 \
  --artifacts-root /path/to/Master_Thesis_Artifacts
```

### 4. Run causal interventions

[`causal_intervention.ipynb`](Main_Experimental_Setup/notebooks/causal_intervention.ipynb)
can run one condition or the complete twelve-condition grid. It compares the
Li, Nanda, and adapted intervention methods and keeps the causal outputs
separate from the common evaluation:

```bash
python scripts/run_causal_intervention.py \
  --architecture transformer \
  --objective ar \
  --board-size 8 \
  --profile smoke \
  --artifacts-root /path/to/Master_Thesis_Artifacts
```

Use the `smoke` profile before a full GPU run. For JEPA, the default `best`
readout is selected only from the validation split; AR uses its native head.

### 5. Aggregate and visualise results

- `comparative_analysis_v3.ipynb` produces the cross-model and cross-board
  comparison.
- `chapter_5_results.ipynb` creates the tables and figures used in the results
  chapter.
- `compute_cost_figures.ipynb` summarises training and evaluation time.
- `jepa_experiments_comparison.ipynb` reconstructs the selected JEPA
  development trajectory.

Ready-to-read outputs are collected in
[`Main_Experimentation_Results`](Main_Experimentation_Results), including the
Chapter 5 tables and figures, the five-seed comparisons, and the Power BI
report definition. Full per-position evaluation dumps and trained weights stay
in the external artifact store.

## Reproducibility and validation

The experiment contract is defined by:

- `canonical_training_protocol.yml`, which fixes the twelve canonical
  training conditions and seed families;
- six JEPA YAML configurations, one for each architecture/board-size pair;
- `thesis_evaluation_registry.yml`, which maps canonical checkpoints and
  corpora into the common evaluator;
- saved hashes and manifests for source, configurations, checkpoints, splits,
  and selected result collections.

Run the standalone repository checks with:

```bash
python -m pytest tests/unit -q
```

The larger `tests/equivalence` suite documents the behavioural comparison made
during the codebase reconstruction. Some of those tests require the preserved
historical source tree and are not needed to use the published package.

The official Mamba construction and BF16 update gate can be repeated in a
compatible CUDA environment with:

```bash
python tools/smoke_official_mamba.py
```

It succeeds only after printing `OFFICIAL_MAMBA_SMOKE_PASS`.

## Scope

This repository is intended to make the thesis's training and evaluation
protocol inspectable and reproducible while keeping multi-gigabyte artifacts
outside version control. It is research software rather than a general-purpose
Othello engine or a competitive game-playing system.
