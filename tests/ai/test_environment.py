from ai.encoding import Action, goal_state
from ai.environment import PuzzleEnvironment


def test_right_moves_blank_and_can_finish():
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, 15)
    env = PuzzleEnvironment(state)

    transition = env.step(Action.RIGHT)

    assert transition.state == goal_state()
    assert transition.done is True
    assert transition.legal is True
    assert transition.reward == 10.0


def test_illegal_move_keeps_state_and_penalizes():
    state = goal_state()
    env = PuzzleEnvironment(state)

    transition = env.step(Action.DOWN)

    assert transition.state == state
    assert transition.done is False
    assert transition.legal is False
    assert transition.reward == -1.0


def test_reset_accepts_a_new_state():
    env = PuzzleEnvironment()
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, 14, 15)

    assert env.reset(state) == state
    assert env.state == state
