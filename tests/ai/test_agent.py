import torch

from ai.agent import (
    AStarGuardedAgent,
    DecisionTrace,
    EnsembleNeuralAgent,
    NeuralAgent,
    NeuralBeamAgent,
    load_agent,
)
from ai.encoding import Action, apply_action, goal_state, legal_actions
from ai.model import HuarongNet, save_checkpoint


class _AlternatingModel:
    def __init__(self):
        self.calls = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, features):
        logits = torch.full((1, 4), -10.0)
        action = Action.DOWN if self.calls % 2 == 0 else Action.UP
        logits[0, int(action)] = 10.0
        self.calls += 1
        return logits, torch.zeros((1, 4))

    def hidden(self, features):
        return torch.zeros((features.shape[0], 64))


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


def test_neural_agent_breaks_immediate_reverse_without_astar():
    state = (4, 5, 6, 10, 11, -1, 1, 13, 9, 12, 3, 8, 7, 14, 2, 15)
    agent = NeuralAgent(_AlternatingModel())

    first = agent.decide(state)
    next_state = apply_action(state, first.selected_action)
    assert next_state is not None
    second = agent.decide(next_state)

    assert first.selected_action == Action.DOWN
    assert second.selected_action != Action.UP
    assert second.decision_source == "neural-anti-loop"
    assert second.fallback_used is False


def test_astar_guard_breaks_a_two_state_loop():
    state = (4, 5, 6, 10, 11, -1, 1, 13, 9, 12, 3, 8, 7, 14, 2, 15)
    down_state = apply_action(state, Action.DOWN)
    assert down_state is not None

    class AlternatingAgent:
        def __init__(self):
            self.actions = iter((Action.DOWN, Action.UP, Action.DOWN))

        def decide(self, state, *, step=0):
            return DecisionTrace(
                state=state,
                blank_position=divmod(state.index(-1), 4),
                candidates=(),
                selected_action=next(self.actions),
                confidence=0.99,
                inference_ms=0.0,
                step=step,
                hidden_summary=(),
            )

    guard = AStarGuardedAgent(
        AlternatingAgent(), confidence_threshold=0.0, agreement_threshold=0.0
    )
    guard.decide(state)
    guard.decide(down_state)
    trace = guard.decide(state)

    assert trace.fallback_used is True
    assert trace.decision_source == "astar-loop-break"
    assert trace.selected_action == Action.UP


def test_load_agent_can_disable_astar_guard_for_ensemble(tmp_path, monkeypatch):
    fold_path = tmp_path / "fold.pt"
    checkpoint_path = tmp_path / "第六代无拐杖数字华容道之神.pt"
    save_checkpoint(fold_path, HuarongNet(), {"stage": "sixth-generation-fold"})
    save_checkpoint(
        checkpoint_path,
        HuarongNet(),
        {
            "stage": "sixth-generation",
            "ensemble_checkpoints": [fold_path.name],
            "astar_guard": True,
        },
    )

    agent, _ = load_agent(checkpoint_path, use_guard=False)

    assert isinstance(agent, EnsembleNeuralAgent)
    assert not isinstance(agent, AStarGuardedAgent)

    monkeypatch.setattr(
        "ai.agent.astar_action",
        lambda state: (_ for _ in ()).throw(AssertionError("A* must not run")),
    )
    trace = agent.decide(goal_state())
    assert trace.fallback_used is False


def test_neural_beam_agent_uses_legal_action_without_astar(monkeypatch):
    """The neural planner must solve a one-move state without an A* call."""
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, 15)
    base_agent = EnsembleNeuralAgent([HuarongNet()])
    agent = NeuralBeamAgent(base_agent, beam_width=4, horizon=4)

    monkeypatch.setattr(
        "ai.agent.astar_action",
        lambda state: (_ for _ in ()).throw(AssertionError("A* must not run")),
    )

    trace = agent.decide(state)

    assert trace.selected_action in legal_actions(state)
    assert trace.selected_action == Action.RIGHT
    assert trace.decision_source == "neural-beam"
    assert trace.fallback_used is False


def test_neural_beam_agent_reset_discards_cached_plan():
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, 14, 15)
    base_agent = EnsembleNeuralAgent([HuarongNet()])
    agent = NeuralBeamAgent(base_agent, beam_width=4, horizon=4)
    plan_calls = 0

    def plan(start):
        nonlocal plan_calls
        plan_calls += 1
        assert start == state
        return (Action.RIGHT, Action.RIGHT)

    agent._plan = plan

    first = agent.decide(state)
    assert first.selected_action == Action.RIGHT
    next_state = apply_action(state, Action.RIGHT)
    assert next_state is not None
    second = agent.decide(next_state)
    assert second.selected_action == Action.RIGHT
    assert plan_calls == 1

    agent.reset()
    agent.decide(state)
    assert plan_calls == 2
