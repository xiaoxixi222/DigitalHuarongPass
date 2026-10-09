from pathlib import Path

import torch

from ai.dataset import generate_examples
from ai.train import _double_dqn_targets, _kfold_examples, run_training


def test_tiny_training_writes_checkpoint_and_report(tmp_path: Path):
    result = run_training(
        samples=24,
        epochs=1,
        eval_games=4,
        seed=3,
        output_root=tmp_path,
    )

    assert result.checkpoint.exists()
    assert result.report.exists()
    assert result.metrics.exists()
    assert "supervised" in result.metrics.read_text()


def test_cross_validation_keeps_states_disjoint():
    examples = generate_examples(20, seed=17, min_depth=2, max_depth=5)
    splits = _kfold_examples(examples, folds=4, seed=17)

    validation_states = [
        example.state for _, validation in splits for example in validation
    ]
    assert len(validation_states) == len(set(validation_states)) == len(examples)
    for train, validation in splits:
        assert {example.state for example in train}.isdisjoint(
            example.state for example in validation
        )


def test_dqn_target_masks_illegal_actions_and_uses_double_dqn_selection():
    rewards = torch.tensor([2.0])
    dones = torch.tensor([0.0])
    # Action 1 has the largest online Q value, but it is illegal. Among legal
    # actions the online network selects action 3, whose target Q is 8.
    online_q = torch.tensor([[1.0, 99.0, 3.0, 4.0]])
    target_q = torch.tensor([[5.0, 50.0, 70.0, 8.0]])
    legal_mask = torch.tensor([[True, False, True, True]])

    targets = _double_dqn_targets(
        rewards,
        dones,
        online_q,
        target_q,
        legal_mask,
        gamma=0.5,
    )

    assert torch.equal(targets, torch.tensor([6.0]))
