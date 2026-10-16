import torch

from ai.encoding import (
    Action,
    ACTION_ORDER,
    ENHANCED_FEATURE_SIZE,
    encode_state,
    encode_state_enhanced,
    goal_state,
    heuristic_score,
    legal_actions,
)


def test_goal_encoding_has_stable_shape_and_dtype():
    encoded = encode_state(goal_state())

    assert encoded.shape == (260,)
    assert encoded.dtype == torch.float32
    assert torch.isfinite(encoded).all()


def test_goal_has_no_legal_moves_and_zero_heuristic():
    assert legal_actions(goal_state()) == (Action.UP, Action.LEFT)
    assert heuristic_score(goal_state()) == 0
    assert ACTION_ORDER == (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def test_encoding_changes_when_two_tiles_are_swapped():
    state = goal_state()
    swapped = state[:-3] + (15, 14, -1)

    assert not torch.equal(encode_state(state), encode_state(swapped))
    assert heuristic_score(swapped) > heuristic_score(state)


def test_enhanced_encoding_is_finite_and_preserves_permutation_signal():
    state = goal_state()
    swapped = state[:-3] + (15, 14, -1)

    encoded = encode_state_enhanced(state)
    swapped_encoded = encode_state_enhanced(swapped)

    assert encoded.shape == (ENHANCED_FEATURE_SIZE,)
    assert encoded.dtype == torch.float32
    assert torch.isfinite(encoded).all()
    assert not torch.equal(encoded, swapped_encoded)
