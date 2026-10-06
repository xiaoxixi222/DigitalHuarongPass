from ai.agent import NeuralAgent
from ai.encoding import Action, goal_state, legal_actions
from ai.model import HuarongNet


def test_agent_never_selects_illegal_action():
    trace = NeuralAgent(HuarongNet()).decide(goal_state(), step=0)

    assert trace.selected_action in legal_actions(goal_state())
    assert len(trace.candidates) == 4
    assert all(0.0 <= candidate.probability <= 1.0 for candidate in trace.candidates)


def test_trace_contains_legal_flags_and_selected_action():
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, 15)
    trace = NeuralAgent(HuarongNet()).decide(state, step=3)

    assert trace.step == 3
    assert trace.selected_action == Action.RIGHT
    assert trace.candidates[Action.DOWN].legal is False
    assert trace.candidates[Action.RIGHT].legal is True
