import torch

from ai.model import HuarongNet, load_checkpoint, save_checkpoint


def test_network_has_two_four_action_heads():
    model = HuarongNet()
    policy, q_values = model(torch.zeros(2, 260))

    assert policy.shape == (2, 4)
    assert q_values.shape == (2, 4)


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "model.pt"
    original = HuarongNet()
    save_checkpoint(path, original, metadata={"stage": "test"})

    restored, metadata = load_checkpoint(path)

    assert metadata["stage"] == "test"
    for left, right in zip(original.parameters(), restored.parameters()):
        assert torch.equal(left, right)
