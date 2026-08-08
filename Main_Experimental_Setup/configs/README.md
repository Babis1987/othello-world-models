# Canonical configuration files

The six JEPA YAML files and `canonical_training_protocol.yml` are immutable
copies of the hash-locked source protocol. Do not edit them to run an ablation;
place non-canonical experiments under `JEPA_Experimentation` instead.

The training notebooks expose board size while asserting their resolved values
against these files before constructing a model.
