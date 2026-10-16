import torch

from ai.model import (
    HuarongNet,
    load_checkpoint,
    load_ensemble_checkpoint,
    save_checkpoint,
    save_ensemble_checkpoint,
)
from ai.encoding import ENHANCED_FEATURE_SIZE


def test_network_has_two_four_action_heads():
    model = HuarongNet()
    policy, q_values = model(torch.zeros(2, 260))

    assert policy.shape == (2, 4)
    assert q_values.shape == (2, 4)


def test_eighth_generation_full_outputs_have_expected_shapes():
    model = HuarongNet()
    outputs = model.forward_full(torch.zeros(3, 260))

    assert outputs["policy_logits"].shape == (3, 4)
    assert outputs["q_values"].shape == (3, 4)
    assert outputs["distance"].shape == (3, 1)
    assert outputs["distance_logits"].shape == (3, 97)
    assert outputs["risk_logits"].shape == (3, 4)


def test_enhanced_model_accepts_v3_state_representation():
    model = HuarongNet(input_size=ENHANCED_FEATURE_SIZE)
    outputs = model.forward_full(torch.zeros(2, ENHANCED_FEATURE_SIZE))

    assert outputs["policy_logits"].shape == (2, 4)
    assert outputs["distance"].shape == (2, 1)


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "model.pt"
    original = HuarongNet()
    save_checkpoint(path, original, metadata={"stage": "test"})

    restored, metadata = load_checkpoint(path)

    assert metadata["stage"] == "test"
    for left, right in zip(original.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_embedded_ensemble_round_trip_does_not_need_fold_files(tmp_path):
    path = tmp_path / "第八代数字华容道高速王.pt"
    models = [HuarongNet(), HuarongNet()]
    save_ensemble_checkpoint(
        path,
        models,
        {"model_name": "第八代数字华容道高速王"},
    )

    loaded, metadata = load_ensemble_checkpoint(path)

    assert len(loaded) == 2
    assert metadata["model_name"] == "第八代数字华容道高速王"
    assert all(model.training is False for model in loaded)
