from ai.dataset import generate_examples, generate_trajectory_examples
from ai.encoding import ACTION_DELTAS, apply_action, goal_state
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
