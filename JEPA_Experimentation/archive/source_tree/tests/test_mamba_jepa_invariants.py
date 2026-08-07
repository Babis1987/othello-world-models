"""CPU invariant tests for the Mamba-JEPA smoke-eval feature-extraction paths.

Diagnostic companion to the mamba_jepa_v1_hd_vicreg audit. Uses a tiny
random-init Mamba encoder (torch backend, 4 layers, d=64) — no checkpoint or
GPU required. The tests pin down the invariants the smoke-eval pipeline relies
on:

    1. Causality: hidden[:, t] depends only on tokens <= t at every layer.
    2. Alignment: hidden[:, t] responds to token t and ignores token t+1, and
       the chunk dataset pairs x[t] with y[t] = token t+1.
    3. Length-1 target prototypes: every vocab token has a distinct, finite
       embedding through the hard-disjoint target path (VICReg variance is
       satisfiable at K=1).
    4. Extraction-path equivalence: the notebook next-move-head path, the
       OthelloJEPA.encode_hidden training path, and the board-probe hook path
       agree — the ONLY expected difference is that the hook path captures the
       residual stream BEFORE the final LayerNorm.
    5. Right-padding invariance: appending pad tokens (as
       evaluate_legal_moves' batching does) must not change earlier positions.
"""

from __future__ import annotations

import pytest
import torch

from othello_research.datasets.dataset import TARGET_PAD, OthelloChunkDataset
from othello_research.datasets.move_mapping import build_mappings
from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_research.probes.board_state import ActivationCache

N_LAYERS = 4
D_MODEL = 64
SEED = 42


@pytest.fixture(scope="module")
def jepa() -> OthelloJEPA:
    torch.manual_seed(SEED)
    cfg = JEPAConfig(
        variant="v1",
        loss_type="vicreg",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=N_LAYERS,
        n_heads=4,
        d_model=D_MODEL,
        dropout=0.0,
        encoder_architecture="mamba",
        d_state=8,
        d_conv=4,  # matches the mamba_jepa_v1_hd_vicreg run
        expand=2,
        mamba_backend="torch",
        predictor_type="linear",
        prediction_horizon=1,
    )
    model = OthelloJEPA(cfg)
    model.eval()
    return model


def _all_layer_hiddens(encoder: torch.nn.Module, x: torch.Tensor) -> dict[int, torch.Tensor]:
    """Capture layer 0 (embedding stream) .. N (block outputs) via the probe hook path."""
    layers = tuple(range(len(encoder.blocks) + 1))
    with torch.no_grad(), ActivationCache(encoder, layers) as cache:
        encoder(x)
        return {layer: cache.activations[layer].clone() for layer in layers}


def _head_path_features(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Replicate the smoke-eval notebook's FrozenJEPANextMoveHead.forward encoder pass."""
    with torch.no_grad():
        pos = torch.arange(x.size(1), device=x.device)
        h = encoder.drop(encoder.wte(x) + encoder.wpe(pos))
        for block in encoder.blocks:
            h = block(h)
        return encoder.ln_f(h)


def test_causality_prefix_hidden_states_match(jepa: OthelloJEPA) -> None:
    """Hidden states at positions <= t must be unaffected by tokens > t."""
    torch.manual_seed(0)
    encoder = jepa.context_encoder
    vocab = jepa.config.vocab_size
    batch, seq_len, t = 4, 20, 11

    x_a = torch.randint(0, vocab, (batch, seq_len))
    x_b = x_a.clone()
    x_b[:, t + 1:] = torch.randint(0, vocab, (batch, seq_len - t - 1))
    assert not torch.equal(x_a, x_b)

    hidden_a = _all_layer_hiddens(encoder, x_a)
    hidden_b = _all_layer_hiddens(encoder, x_b)
    for layer, h_a in hidden_a.items():
        delta = (h_a[:, : t + 1] - hidden_b[layer][:, : t + 1]).abs().max().item()
        assert delta < 1e-6, (
            f"Causality violated at layer {layer}: prefix hidden states moved "
            f"by {delta} when only tokens > {t} changed."
        )
    # VERDICT is asserted, printed for the audit log:
    print("VERDICT: PASS — hidden[:, <=t] independent of future tokens at every layer")


def test_alignment_hidden_t_tracks_token_t_not_t_plus_1(jepa: OthelloJEPA) -> None:
    """hidden[:, t] must change with token t and ignore token t+1."""
    torch.manual_seed(1)
    encoder = jepa.context_encoder
    vocab = jepa.config.vocab_size
    batch, seq_len, t = 4, 16, 7

    x = torch.randint(0, vocab, (batch, seq_len))
    x_diff_t = x.clone()
    x_diff_t[:, t] = (x[:, t] + 1) % vocab
    x_diff_next = x.clone()
    x_diff_next[:, t + 1] = (x[:, t + 1] + 1) % vocab

    base = _all_layer_hiddens(encoder, x)
    diff_t = _all_layer_hiddens(encoder, x_diff_t)
    diff_next = _all_layer_hiddens(encoder, x_diff_next)

    for layer in base:
        moved = (base[layer][:, t] - diff_t[layer][:, t]).abs().max().item()
        assert moved > 1e-6, (
            f"Layer {layer}: hidden[:, {t}] did not respond to a change in token {t}."
        )
        frozen = (base[layer][:, t] - diff_next[layer][:, t]).abs().max().item()
        assert frozen < 1e-6, (
            f"Layer {layer}: hidden[:, {t}] leaked information from token {t + 1} "
            f"(max delta {frozen}) — off-by-one in conv padding or scan direction."
        )
    print("VERDICT: PASS — hidden[:, t] is a function of tokens <= t and reacts to token t")


def test_dataset_pairs_hidden_t_with_token_t_plus_1() -> None:
    """OthelloChunkDataset must supervise position t with token t+1 (TARGET_PAD elsewhere)."""
    board_size = 8
    raw_to_token, token_to_raw = build_mappings(board_size)
    playable = [raw for raw, tok in enumerate(raw_to_token) if tok != -1]
    game = playable[:10]
    block_size = board_size * board_size - 5

    dataset = OthelloChunkDataset(games=[game], block_size=block_size, board_size=board_size)
    x, y = dataset[0]
    tokens = [raw_to_token[raw] for raw in game]

    assert x[: len(game) - 1].tolist() == tokens[:-1]
    assert y[: len(game) - 1].tolist() == tokens[1:], "y[t] must be token t+1"
    assert (y[len(game) - 1 :] == TARGET_PAD).all(), "padded targets must be TARGET_PAD"
    input_pad = board_size * board_size - 4
    assert (x[len(game) :] == input_pad).all(), "padded inputs must use the input pad token"
    print("VERDICT: PASS — dataset supervision is hidden[:, t] -> token t+1 with TARGET_PAD")


def test_length1_target_prototypes_are_distinct(jepa: OthelloJEPA) -> None:
    """Every vocab token encoded as a length-1 target sequence gets a distinct embedding."""
    encoder = jepa.context_encoder
    vocab = jepa.config.vocab_size
    move_tokens = torch.arange(vocab - 1)  # exclude the input pad token
    absolute_position = 10  # hard-disjoint targets keep absolute game positions

    idx = move_tokens.unsqueeze(1)  # (V, 1) length-1 sequences
    positions = torch.full_like(idx, absolute_position)
    with torch.no_grad():
        prototypes = jepa.encode_hidden(encoder, idx, positions)[:, -1, :]

    assert torch.isfinite(prototypes).all(), "length-1 target path produced non-finite values"
    distances = torch.cdist(prototypes, prototypes)
    off_diag = distances[~torch.eye(len(move_tokens), dtype=torch.bool)]
    assert (off_diag > 0).all(), "two vocab tokens collapsed to the same target prototype"
    per_dim_std = prototypes.std(dim=0)
    assert (per_dim_std > 0).all(), "some embedding dimension is constant across tokens"
    print(
        "VERDICT: PASS — length-1 target prototypes distinct "
        f"(min pairwise dist {off_diag.min():.4f}, mean per-dim std {per_dim_std.mean():.4f})"
    )


def test_extraction_paths_agree_up_to_final_layernorm(jepa: OthelloJEPA) -> None:
    """Head path == training path == ln_f(hook path); raw hook path differs by ln_f only."""
    torch.manual_seed(2)
    encoder = jepa.context_encoder
    vocab = jepa.config.vocab_size
    x = torch.randint(0, vocab, (4, 16))

    head_features = _head_path_features(encoder, x)
    with torch.no_grad():
        training_features = jepa.encode_hidden(encoder, x)
    hook_last_block = _all_layer_hiddens(encoder, x)[N_LAYERS]

    delta_train = (head_features - training_features).abs().max().item()
    assert delta_train < 1e-5, (
        f"Notebook head path and OthelloJEPA.encode_hidden diverge (max delta {delta_train})."
    )

    with torch.no_grad():
        hook_normed = encoder.ln_f(hook_last_block)
    delta_hook = (head_features - hook_normed).abs().max().item()
    assert delta_hook < 1e-5, (
        f"Head path != ln_f(board-probe hook path) (max delta {delta_hook}) — "
        "the two eval consumers see different features beyond the documented ln_f gap."
    )

    raw_gap = (head_features - hook_last_block).abs().max().item()
    assert raw_gap > 1e-4, (
        "Expected the raw hook features to differ from the head features by ln_f; "
        "if they are identical, ln_f is degenerate."
    )
    print(
        "VERDICT: PASS — paths agree; documented difference: board-probe hooks capture the "
        f"residual stream pre-ln_f (raw gap {raw_gap:.4f}), heads consume post-ln_f features"
    )


def test_right_padding_does_not_change_scored_positions(jepa: OthelloJEPA) -> None:
    """evaluate_legal_moves right-pads batches; earlier positions must be unaffected."""
    torch.manual_seed(3)
    encoder = jepa.context_encoder
    vocab = jepa.config.vocab_size
    pad_token = vocab - 1
    batch, true_len, padded_len = 4, 9, 18

    x_short = torch.randint(0, vocab - 1, (batch, true_len))
    x_padded = torch.full((batch, padded_len), pad_token, dtype=torch.long)
    x_padded[:, :true_len] = x_short

    hidden_short = _all_layer_hiddens(encoder, x_short)
    hidden_padded = _all_layer_hiddens(encoder, x_padded)
    for layer, h_short in hidden_short.items():
        delta = (h_short - hidden_padded[layer][:, :true_len]).abs().max().item()
        assert delta < 1e-5, (
            f"Layer {layer}: right padding changed scored positions by {delta}."
        )
    print("VERDICT: PASS — right padding leaves scored positions unchanged at every layer")
