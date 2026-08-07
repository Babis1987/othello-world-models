# Master Thesis Code — Final Research Repository

This repository presents the Othello world-model experiments in a clean,
reproducible structure while preserving the exact behaviour of the original
research code in the sibling `Master_Thesis_Code` archive.

## Repository sections

- `JEPA_Experimentation/` documents the full path used to select the final JEPA
  formulation, including training, evaluation, and diagnostic experiments.
- `Main_Experimental_Setup/` contains the canonical 2 × 2 comparison:
  Transformer/Mamba × AR/JEPA over 8 × 8, 12 × 12, and 16 × 16 boards.
- `src/othello_thesis/` contains the shared game engine, data, model, training,
  evaluation, and reporting implementation.
- `tests/` proves compatibility with the original working-tree implementation.

The final JEPA method used in the main experiment is v5 hard-disjoint,
all-position contrastive InfoNCE with an EMA target encoder.

## Preservation contract

Refactoring in this repository must not alter training behaviour, evaluation
semantics, or result presentation. See [AGENTS.md](AGENTS.md) and
[EQUIVALENCE_MATRIX.md](EQUIVALENCE_MATRIX.md) for the enforced contract.

## Current status

The repository is being reconstructed incrementally. The authoritative resume
point is [REFACTOR_STATUS.md](REFACTOR_STATUS.md).

