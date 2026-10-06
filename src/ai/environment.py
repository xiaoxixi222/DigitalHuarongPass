from __future__ import annotations

from dataclasses import dataclass

from .encoding import (
    Action,
    PuzzleState,
    apply_action,
    goal_state,
    heuristic_score,
)


@dataclass(frozen=True)
class Transition:
    state: PuzzleState
    reward: float
    done: bool
    legal: bool
    heuristic_before: int
    heuristic_after: int


class PuzzleEnvironment:
    def __init__(self, initial_state: PuzzleState | None = None, max_steps: int = 200):
        self.max_steps = max_steps
        self.state = goal_state()
        self.step_count = 0
        self.reset(initial_state)

    def reset(self, state: PuzzleState | None = None) -> PuzzleState:
        self.state = goal_state() if state is None else tuple(state)
        self.step_count = 0
        heuristic_score(self.state)
        return self.state

    def step(self, action: Action) -> Transition:
        before = heuristic_score(self.state)
        next_state = apply_action(self.state, action)
        if next_state is None:
            return Transition(self.state, -1.0, False, False, before, before)

        self.state = next_state
        self.step_count += 1
        after = heuristic_score(next_state)
        done = next_state == goal_state() or self.step_count >= self.max_steps
        reward = 10.0 if next_state == goal_state() else 0.1 * (before - after) - 0.01
        return Transition(next_state, reward, done, True, before, after)
