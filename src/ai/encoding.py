from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

import torch

BOARD_SIDE = 4
BOARD_SIZE = BOARD_SIDE * BOARD_SIDE
PuzzleState: TypeAlias = tuple[int, ...]


class Action(IntEnum):
    """A blank-tile movement direction."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_ORDER = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


def goal_state() -> PuzzleState:
    return tuple(range(1, BOARD_SIZE)) + (-1,)


def validate_state(state: PuzzleState) -> None:
    if len(state) != BOARD_SIZE or set(state) != set(range(1, BOARD_SIZE)) | {-1}:
        raise ValueError("state must contain 1..15 and one blank (-1)")


def blank_position(state: PuzzleState) -> tuple[int, int]:
    validate_state(state)
    index = state.index(-1)
    return divmod(index, BOARD_SIDE)


def apply_action(state: PuzzleState, action: Action) -> PuzzleState | None:
    """Apply a blank movement, returning None for an out-of-bounds action."""
    validate_state(state)
    try:
        action = Action(action)
    except ValueError as exc:
        raise ValueError(f"unknown action: {action!r}") from exc

    blank_row, blank_col = blank_position(state)
    delta_row, delta_col = ACTION_DELTAS[action]
    target_row = blank_row + delta_row
    target_col = blank_col + delta_col
    if not (0 <= target_row < BOARD_SIDE and 0 <= target_col < BOARD_SIDE):
        return None

    blank_index = blank_row * BOARD_SIDE + blank_col
    target_index = target_row * BOARD_SIDE + target_col
    next_state = list(state)
    next_state[blank_index], next_state[target_index] = (
        next_state[target_index],
        next_state[blank_index],
    )
    return tuple(next_state)


def legal_actions(state: PuzzleState) -> tuple[Action, ...]:
    return tuple(action for action in ACTION_ORDER if apply_action(state, action) is not None)


def heuristic_score(state: PuzzleState) -> int:
    """Return Manhattan distance plus linear conflicts."""
    validate_state(state)
    distance = 0
    for index, value in enumerate(state):
        if value == -1:
            continue
        row, col = divmod(index, BOARD_SIDE)
        goal_row, goal_col = divmod(value - 1, BOARD_SIDE)
        distance += abs(row - goal_row) + abs(col - goal_col)

    conflicts = 0
    for row in range(BOARD_SIDE):
        target_cols = [
            (state[row * BOARD_SIDE + col] - 1) % BOARD_SIDE
            for col in range(BOARD_SIDE)
            if state[row * BOARD_SIDE + col] != -1
            and (state[row * BOARD_SIDE + col] - 1) // BOARD_SIDE == row
        ]
        conflicts += sum(
            1
            for left in range(len(target_cols))
            for right in range(left + 1, len(target_cols))
            if target_cols[left] > target_cols[right]
        )

    for col in range(BOARD_SIDE):
        target_rows = [
            (state[row * BOARD_SIDE + col] - 1) // BOARD_SIDE
            for row in range(BOARD_SIDE)
            if state[row * BOARD_SIDE + col] != -1
            and (state[row * BOARD_SIDE + col] - 1) % BOARD_SIDE == col
        ]
        conflicts += sum(
            1
            for top in range(len(target_rows))
            for bottom in range(top + 1, len(target_rows))
            if target_rows[top] > target_rows[bottom]
        )
    return distance + 2 * conflicts


def encode_state(state: PuzzleState) -> torch.Tensor:
    """Encode a state into a fixed 260-value float tensor."""
    validate_state(state)
    values: list[float] = []
    for tile in state:
        index = 0 if tile == -1 else tile
        values.extend(1.0 if position == index else 0.0 for position in range(BOARD_SIZE))

    blank_row, blank_col = blank_position(state)
    manhattan = sum(
        abs(row - divmod(value - 1, BOARD_SIDE)[0])
        + abs(col - divmod(value - 1, BOARD_SIDE)[1])
        for row in range(BOARD_SIDE)
        for col in range(BOARD_SIDE)
        for value in (state[row * BOARD_SIDE + col],)
        if value != -1
    )
    conflicts = max(0, heuristic_score(state) - manhattan)
    values.extend(
        (
            blank_row / (BOARD_SIDE - 1),
            blank_col / (BOARD_SIDE - 1),
            manhattan / 48.0,
            conflicts / 24.0,
        )
    )
    return torch.tensor(values, dtype=torch.float32)
