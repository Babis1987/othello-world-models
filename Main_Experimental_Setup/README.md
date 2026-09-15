# Main Experimental Setup

This section contains the canonical 2 × 2 thesis comparison across 8 × 8,
12 × 12, and 16 × 16 Othello boards.

| Architecture | Objective | Canonical training frontend |
| --- | --- | --- |
| Transformer | AR | `notebooks/train_transformer_ar.ipynb` |
| Mamba | AR | `notebooks/train_mamba_ar.ipynb` |
| Transformer | JEPA | `notebooks/train_transformer_jepa.ipynb` |
| Mamba | JEPA | `notebooks/train_mamba_jepa.ipynb` |

The notebooks resolve from the locked protocol rather than copying historical
selectors. Each notebook fixes its architecture/objective and exposes
`BOARD_SIZE` (`8`, `12`, or `16`) at the beginning.

`BOARD_SIZE` defaults to 12. Each notebook uses one visible model factory that
the trainer invokes exactly once after seeding, so inspecting the model does
not perturb the RNG sequence used by initialization or DataLoader shuffling.

Each notebook also exposes `SEED`, so every member of the locked seed family
can be trained directly with the same canonical implementation. The source
research repository's automated multi-seed scheduler, completion-manifest
validator, environment-lock writer, and family reporter are not duplicated
here. Consequently, `default_additional_seeds` in the locked protocol records
the thesis replication plan; it does not automatically launch those runs from
this repository.

## Evaluation frontends

- `notebooks/common_evaluation.ipynb` runs the shared legal-move, frozen-head,
  board-probe, summary, and comparison protocol. It does not run causal
  interventions. Source drift is rejected by default; its visible override is
  only for deliberate reuse of known completed pre-refactor stages.
- `notebooks/causal_intervention.ipynb` independently selects Architecture,
  Objective, and board size and compares the Li et al., Nanda et al., and
  thesis-adapted interventions. The adapted method's current primary random
  control preserves cross-layer probe geometry; the earlier independent-layer
  control is retained only as sensitivity analysis. It writes below the
  selected run's separate `causal_intervention/` tree and never rewrites
  existing common evaluation results. Its last cell reads the saved JSON and
  displays all three methods and their counterfactual deltas without rerunning
  the model.
- `notebooks/comparative_analysis_v3.ipynb` builds the controlled comparison
  reports from completed common evaluations.

## Final JEPA method

The main experiment uses the generic `OthelloJEPA` topology with:

- independently constructed context and target encoders;
- context weights copied into the frozen EMA target encoder;
- a linear 512 → 512 predictor;
- hard-disjoint one-token targets at their original absolute positions;
- every valid prediction boundary;
- position-stratified in-batch InfoNCE at temperature `0.1`;
- pair-count-weighted reductions and EMA momentum `0.996`.

This is not the older prefix-grouped implementation stored in
`JEPA_Experimentation`.

## Source of truth

- `configs/canonical_training_protocol.yml` locks the common AR settings,
  model geometry, data roots, seeds, package versions, and source hashes.
- The six JEPA YAML files are byte-identical to the canonical source configs.
- `configs/thesis_evaluation_registry.yml` identifies the canonical evaluation
  cases and output locations.
- `reference_contracts/` stores resolved, machine-checkable expectations.

Large corpora, checkpoints, and evaluation artifacts remain outside this code
repository.
