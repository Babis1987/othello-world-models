from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import torch

from othello_research.evaluation.linear_head import JEPALinearHead
from othello_research.objectives.jepa_order_aware import (
    JEPAOrderAwareConfig,
    OthelloJEPAOrderAware,
)
from othello_research.training import train_jepa as tjm


NOTEBOOKS = (
    "othello_gpt_jepa_training.ipynb",
    "othello_gpt_jepa_smoke_evaluation.ipynb",
    "othello_gpt_jepa_evaluation_v2.ipynb",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _notebook(name: str) -> dict:
    path = _repo_root() / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(name: str, marker: str) -> str:
    matches = [
        "".join(cell.get("source", []))
        for cell in _notebook(name)["cells"]
        if marker in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1, (name, marker, len(matches))
    return matches[0]


def _tiny_checkpoint() -> tuple[dict, OthelloJEPAOrderAware]:
    cfg = JEPAOrderAwareConfig(
        board_size=8,
        n_layers=1,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        predictor_dropout=0.0,
        action_dim=16,
    )
    model = OthelloJEPAOrderAware(cfg)
    checkpoint = {
        "model": model.state_dict(),
        "model_config": {"objective_class": "jepa_order_aware", **asdict(cfg)},
        "jepa_config": asdict(cfg),
        "train_config": {
            "objective_class": "jepa_order_aware",
            "variant": "v8",
        },
        "step": 1,
        "games_seen": 8,
    }
    return checkpoint, model


def test_target_notebook_code_cells_compile() -> None:
    for name in NOTEBOOKS:
        for index, cell in enumerate(_notebook(name)["cells"]):
            if cell.get("cell_type") != "code":
                continue
            cell_source = "".join(cell.get("source", []))
            if cell_source.lstrip().startswith(("!", "%")):
                continue
            compile(cell_source, f"{name}:cell{index}", "exec")


def test_training_notebook_v8_sanity_cell_executes() -> None:
    cell_source = _cell_source(
        "othello_gpt_jepa_training.ipynb",
        "elif objective_name == 'jepa_order_aware':",
    )
    script = (
        "from pathlib import Path\n"
        "import torch\n"
        f"CONFIG_PATH = Path({str(_repo_root() / 'configs' / 'jepa_v8_order_aware.yml')!r})\n"
        f"DATA_DIR = Path({str(_repo_root() / '__unused_data__')!r})\n"
        f"exec(compile({cell_source!r}, 'v8-training-sanity', 'exec'))\n"
        "assert objective_name == 'jepa_order_aware'\n"
        "assert torch.isfinite(out['loss'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_smoke_and_v2_context_evaluation_paths_load_v8_checkpoint() -> None:
    checkpoint, _ = _tiny_checkpoint()
    loaded, cfg = tjm.build_model_from_checkpoint(checkpoint)
    loaded.load_state_dict(checkpoint["model"])
    loaded.eval()
    assert cfg.variant == "v8"
    assert hasattr(loaded, "context_encoder")

    tokens = torch.randint(0, loaded.config.vocab_size, (2, 6))
    hidden = loaded.encode_hidden(loaded.context_encoder, tokens)
    assert hidden.shape == (2, 6, loaded.config.d_model)

    head = JEPALinearHead(loaded)
    logits, loss = head(tokens, tokens)
    assert logits.shape == (2, 6, loaded.config.vocab_size)
    assert torch.isfinite(loss)


def test_evaluation_v2_skips_action_conditioned_predictor_space_cleanly() -> None:
    checkpoint, _ = _tiny_checkpoint()
    loaded, _ = tjm.build_model_from_checkpoint(checkpoint)
    loaded.load_state_dict(checkpoint["model"])
    helpers_source = _cell_source(
        "othello_gpt_jepa_evaluation_v2.ipynb",
        "# Predictor-space helpers. These do not modify or fine-tune JEPA.",
    )
    namespace = {
        "jepa_model": loaded,
        "device": torch.device("cpu"),
    }
    exec(compile(helpers_source, "predictor-space-helpers", "exec"), namespace)
    assert namespace["PREDICTOR_SPACE_SUPPORTED"] is False

    next_move_source = _cell_source(
        "othello_gpt_jepa_evaluation_v2.ipynb",
        "Predictor-space next-move evaluation skipped:",
    )
    board_source = _cell_source(
        "othello_gpt_jepa_evaluation_v2.ipynb",
        "Predictor-space board probes skipped:",
    )
    exec(compile(next_move_source, "predictor-next-move-skip", "exec"), namespace)
    exec(compile(board_source, "predictor-board-skip", "exec"), namespace)
    assert namespace["predictor_next_move_results"] == {}
    assert namespace["predictor_next_move_models"] == {}
    assert namespace["predictor_board_results"] == {}


def test_training_notebook_prepares_v8_offline_index_before_training() -> None:
    source = _cell_source(
        "othello_gpt_jepa_training.ipynb",
        "Building the efficient v8 pair index on local disk:",
    )
    assert "tools' / 'build_v8_pair_index.py" in source
    assert "order_index_positions_per_game" in source
    assert "expected_constructive" in source
    assert "use_constructive_pairs" in source
    assert "expected_surface_hard_negatives" in source
    assert "order_index_use_surface_hard_negatives" in source
    assert "surface_jaccard_threshold" in source
    assert "surface_max_pairs_per_anchor" in source
    assert "use_surface_hard_negatives" in source
    assert "PAIR_INDEX_RUNTIME_PATH" in source
    assert "shutil.copy2(PAIR_INDEX_RUNTIME_PATH, PAIR_INDEX_PATH)" in source
    # The build now streams output line-by-line via Popen so the per-chunk
    # progress lines show up live in the cell output instead of being held
    # in subprocess stdout's block buffer until the whole build completes.
    assert "run_build_streaming(command)" in source
    assert "subprocess.Popen(" in source
