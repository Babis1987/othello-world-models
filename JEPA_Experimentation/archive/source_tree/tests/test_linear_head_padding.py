import torch

from othello_research.datasets.dataset import TARGET_PAD
from othello_research.evaluation.linear_head import JEPALinearHead
from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA


def test_jepa_linear_head_all_pad_targets_has_finite_zero_loss() -> None:
    torch.manual_seed(0)
    jepa = OthelloJEPA(
        JEPAConfig(
            board_size=8,
            n_layers=1,
            n_heads=4,
            d_model=32,
            dropout=0.0,
            predictor_type="linear",
            prediction_horizon=1,
        )
    )
    head = JEPALinearHead(jepa)
    idx = torch.randint(0, head.config.vocab_size, (3, 5))
    targets = torch.full_like(idx, TARGET_PAD)
    logits, loss = head(idx, targets)
    assert logits.shape == (3, 5, head.config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in head.head.parameters()
    )
