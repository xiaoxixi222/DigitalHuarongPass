from ai.dataset import (
    augment_example,
    collect_dagger_examples,
    generate_examples,
    generate_uniform_examples,
    generate_uniform_trajectory_examples,
    generate_trajectory_examples,
)
from ai.agent import NeuralAgent
from ai.encoding import ACTION_DELTAS, Action, apply_action, goal_state
from ai.model import HuarongNet
from board import Board
from board.board import END


def _replay_teacher_route(example):
    board = Board(4, 4)
    board.board = [
        list(example.state[row * 4 : (row + 1) * 4]) for row in range(4)
    ]
    road = list(board.get_road(weight=2.0))
    assert road and road[-1] is END
    direction_to_action = {
        (delta_col, delta_row): action
        for action, (delta_row, delta_col) in ACTION_DELTAS.items()
    }
    actions = [direction_to_action[direction] for direction in road[:-1]]
    assert actions
    assert actions[0] == example.expert_action
    assert len(actions) == example.remaining_steps

    state = example.state
    for action in actions:
        state = apply_action(state, action)
        assert state is not None
    return state


def test_dataset_is_reproducible():
    first = generate_examples(12, seed=7, min_depth=2, max_depth=5)
    second = generate_examples(12, seed=7, min_depth=2, max_depth=5)

    assert first == second
    assert len(first) == 12
    assert all(example.remaining_steps > 0 for example in first)


def test_trajectory_examples_are_unique_and_replay_to_goal():
    first = generate_trajectory_examples(24, seed=19, min_depth=4, max_depth=8)
    second = generate_trajectory_examples(24, seed=19, min_depth=4, max_depth=8)

    assert first == second
    assert len(first) == 24
    assert len({example.state for example in first}) == 24
    assert all(_replay_teacher_route(example) == goal_state() for example in first)


def test_trajectory_examples_support_long_training_walks():
    examples = generate_trajectory_examples(
        8,
        seed=20261008,
        min_depth=900,
        max_depth=1000,
    )

    assert len(examples) == 8
    assert len({example.state for example in examples}) == 8
    assert all(_replay_teacher_route(example) == goal_state() for example in examples)


def test_uniform_examples_are_unique_and_replayable():
    examples = generate_uniform_examples(12, seed=23)

    assert len(examples) == 12
    assert len({example.state for example in examples}) == 12
    assert all(apply_action(example.state, example.expert_action) is not None for example in examples)
    assert all(_replay_teacher_route(example) == goal_state() for example in examples)


def test_uniform_trajectory_examples_cover_full_routes():
    examples = generate_uniform_trajectory_examples(24, seed=31)

    assert len(examples) == 24
    assert max(example.remaining_steps for example in examples) > 1
    assert all(apply_action(example.state, example.expert_action) is not None for example in examples)


def test_horizontal_reflection_swaps_left_and_right_labels():
    from ai.dataset import Example

    example = Example(goal_state(), Action.LEFT, 1)
    augmented = augment_example(example)

    assert len(augmented) == 8
    assert augmented[4].expert_action == Action.RIGHT


def test_dagger_collection_returns_unique_teacher_labeled_states():
    start = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, 14, 15)
    examples = collect_dagger_examples(
        NeuralAgent(HuarongNet()),
        [start],
        max_steps=4,
        seed=5,
    )

    assert examples
    assert len({example.state for example in examples}) == len(examples)
    assert all(apply_action(example.state, example.expert_action) is not None for example in examples)
