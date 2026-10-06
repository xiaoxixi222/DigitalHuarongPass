from __future__ import annotations

import random
from dataclasses import dataclass

from board import Board
from board.board import END

from .encoding import (
    ACTION_DELTAS,
    ACTION_ORDER,
    Action,
    PuzzleState,
    apply_action,
    goal_state,
)


@dataclass(frozen=True)
class Example:
    state: PuzzleState
    expert_action: Action
    remaining_steps: int


def _random_walk_states(
    count: int,
    *,
    seed: int,
    min_depth: int,
    max_depth: int,
) -> list[PuzzleState]:
    rng = random.Random(seed)
    states: list[PuzzleState] = []
    seen: set[PuzzleState] = {goal_state()}
    attempts = 0
    while len(states) < count and attempts < count * 80:
        attempts += 1
        state = goal_state()
        previous: Action | None = None
        depth = rng.randint(min_depth, max_depth)
        for _ in range(depth):
            candidates = [
                action
                for action in ACTION_ORDER
                if apply_action(state, action) is not None
                and not (
                    previous is not None
                    and (int(action) ^ 1) == int(previous)
                )
            ]
            action = rng.choice(candidates or list(ACTION_ORDER))
            state = apply_action(state, action)
            assert state is not None
            previous = action
        if state not in seen:
            seen.add(state)
            states.append(state)
    if len(states) < count:
        raise RuntimeError(f"could only generate {len(states)} of {count} unique states")
    return states


def _label_state(state: PuzzleState) -> Example | None:
    board = Board(4, 4)
    board.board = [list(state[row * 4 : (row + 1) * 4]) for row in range(4)]
    road = list(board.get_road(weight=2.0))
    if not road or road[0] is END or not isinstance(road[0], tuple):
        return None
    direction = road[0]
    direction_to_action = {
        (delta_col, delta_row): action
        for action, (delta_row, delta_col) in ACTION_DELTAS.items()
    }
    action = direction_to_action.get(direction)
    if action is None:
        raise ValueError(f"unknown A* direction: {direction!r}")
    return Example(state, action, len(road) - 1)


def generate_examples(
    count: int,
    *,
    seed: int = 42,
    min_depth: int = 2,
    max_depth: int = 12,
) -> tuple[Example, ...]:
    if count <= 0:
        return ()
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError("depth range must satisfy 1 <= min_depth <= max_depth")
    examples: list[Example] = []
    for state in _random_walk_states(
        count,
        seed=seed,
        min_depth=min_depth,
        max_depth=max_depth,
    ):
        labeled = _label_state(state)
        if labeled is not None:
            examples.append(labeled)
    if len(examples) < count:
        raise RuntimeError(f"could only label {len(examples)} of {count} states")
    return tuple(examples)
