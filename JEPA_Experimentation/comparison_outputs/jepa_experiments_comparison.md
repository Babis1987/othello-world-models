| Στάδιο | Αρχιτεκτονική | Πείραμα | Top-1 νόμιμη κίνηση | Readout | Σχετική κατάσταση ταμπλό | Probe / layer | Παιχνίδια εκπαίδευσης | Παιχνίδια αξιολόγησης | Πρωτόκολλο |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Κοινό ιστορικό (nested) | Transformer | v1 — VICReg | 25.7% | LINEAR | 62.4% | LINEAR / L0 | 19,999,840 | 5,000 | legacy evaluation |
| Κοινό ιστορικό (nested) | Transformer | v2 — Smooth L1, EMA, K=4 | 38.6% | LINEAR | 62.0% | LINEAR / L6 | 19,999,840 | 5,000 | legacy evaluation |
| Κοινό ιστορικό (nested) | Transformer | v3 — Smooth L1, EMA, K=8 (collapse) | 27.2% | LINEAR | 59.7% | LINEAR / L0 | 19,999,840 | 5,000 | legacy evaluation |
| Κοινό ιστορικό (nested) | Transformer | v4 — Smooth L1, multi-position K=4 | 33.6% | LINEAR | 62.7% | LINEAR / L1 | 19,999,840 | 5,000 | legacy evaluation |
| Κοινό ιστορικό (nested) | Transformer | v5 — InfoNCE | 39.6% | LINEAR | 63.0% | LINEAR / L5 | 19,999,840 | 5,000 | legacy evaluation |
| v6 hard-disjoint δράση | Transformer | v6 — Distance-margin, 10 modes | 62.2% | MLP | 63.1% | MLP / L0 | 4,999,952 | 5,000 | development evaluation |
| v6 hard-disjoint δράση | Mamba | v6 — Distance-margin, 10 modes | 57.1% | MLP | 61.7% | MLP / L0 | 19,999,840 | 5,000 | smoke evaluation |
| Hard-disjoint | Transformer | v1 — VICReg | 79.1% | MLP | 70.3% | LINEAR / L5 | 19,999,840 | 5,000 | normalized_v2 |
| Hard-disjoint | Transformer | v2 — Smooth L1, EMA, K=4 | 37.8% | MLP | 62.0% | MLP / L0 | 4,999,952 | 5,000 | smoke evaluation |
| Hard-disjoint | Transformer | v4 — Smooth L1, multi-position K=4 | 25.3% | MLP | 60.5% | MLP / L0 | 4,999,956 | 5,000 | smoke evaluation |
| Hard-disjoint | Transformer | v5 — InfoNCE | 65.7% | MLP | 62.9% | MLP / L0 | 19,999,840 | 5,000 | development evaluation |
| Hard-disjoint | Mamba | v1 — VICReg | 94.3% | LINEAR | 84.4% | LINEAR / L7 | 19,999,840 | 5,000 | normalized_v2 |
| Hard-disjoint | Mamba | v5 — InfoNCE | 54.4% | MLP | 65.2% | MLP / L0 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Transformer | v1 — VICReg | 99.0% | LINEAR | 92.3% | LINEAR / L2 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Transformer | v2 — Smooth L1, EMA, K=4 | 98.9% | LINEAR | 90.8% | LINEAR / L2 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Transformer | v4 — Smooth L1, multi-position K=4 | 97.7% | MLP | 91.8% | LINEAR / L3 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Transformer | v5 — InfoNCE | 99.4% | LINEAR | 97.0% | LINEAR / L6 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Mamba | v1 — VICReg | 99.5% | LINEAR | 96.1% | LINEAR / L7 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Mamba | v1e — VICReg + EMA | 99.7% | LINEAR | 97.1% | LINEAR / L7 | 19,999,840 | 5,000 | unified_eval_v2 |
| Hard-disjoint + all-position | Mamba | v5 — InfoNCE | 99.7% | LINEAR | 98.2% | LINEAR / L12 | 19,999,840 | 5,000 | unified_eval_v2 |
