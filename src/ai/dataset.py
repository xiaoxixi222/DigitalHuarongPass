from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from board import Board
from board.board import END, is_solvable

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
    next_state: PuzzleState | None = None
    distance_bucket: int | None = None
    risk_target: tuple[float, ...] | None = None


_DIRECTION_TO_ACTION = {
    (delta_col, delta_row): action
    for action, (delta_row, delta_col) in ACTION_DELTAS.items()
}


def _random_walk_state(
    rng: random.Random,
    *,
    min_depth: int,
    max_depth: int,
) -> PuzzleState:
    """Sample one solvable state by walking backwards from the goal."""
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
    return state


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
        state = _random_walk_state(
            rng,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if state not in seen:
            seen.add(state)
            states.append(state)
    if len(states) < count:
        raise RuntimeError(f"could only generate {len(states)} of {count} unique states")
    return states


def _teacher_trajectory(
    state: PuzzleState,
) -> tuple[tuple[PuzzleState, Action, int], ...] | None:
    """Return ``(state, action, remaining_steps)`` entries for one A* route.

    A* is deliberately confined to this offline dataset module.  The route is
    replayed through the shared ``apply_action`` implementation so labels use
    the same action semantics as the runtime agent.
    """
    board = Board(4, 4)
    board.board = [list(state[row * 4 : (row + 1) * 4]) for row in range(4)]
    road = list(board.get_road(weight=2.0))
    if not road or road[0] is END or not isinstance(road[0], tuple):
        return None

    directions = road[:-1] if road[-1] is END else road
    if not directions:
        return None

    entries: list[tuple[PuzzleState, Action, int]] = []
    current = state
    for index, direction in enumerate(directions):
        action = _DIRECTION_TO_ACTION.get(direction)
        if action is None:
            raise ValueError(f"unknown A* direction: {direction!r}")
        next_state = apply_action(current, action)
        if next_state is None:
            raise ValueError(
                f"A* produced illegal direction {direction!r} for state {current!r}"
            )
        entries.append((current, action, len(directions) - index))
        current = next_state

    if current != goal_state():
        return None
    return tuple(entries)


def _label_state(state: PuzzleState) -> Example | None:
    trajectory = _teacher_trajectory(state)
    if not trajectory:
        return None
    route_state, action, remaining_steps = trajectory[0]
    next_state = apply_action(route_state, action)
    return Example(
        route_state,
        action,
        remaining_steps,
        next_state=next_state,
        distance_bucket=min(96, remaining_steps),
    )


_ACTION_VECTOR = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}
_VECTOR_ACTION = {vector: action for action, vector in _ACTION_VECTOR.items()}


def _transform_state(
    state: PuzzleState,
    transform: int,
) -> tuple[PuzzleState, dict[Action, Action]]:
    """Apply one of the eight board symmetries and transform actions with it."""
    values = [[state[row * 4 + col] for col in range(4)] for row in range(4)]

    def map_position(row: int, col: int) -> tuple[int, int]:
        for _ in range(transform % 4):
            row, col = col, 3 - row
        if transform >= 4:
            col = 3 - col
        return row, col

    transformed = [-1] * 16
    for row in range(4):
        for col in range(4):
            target_row, target_col = map_position(row, col)
            transformed[target_row * 4 + target_col] = values[row][col]

    action_map: dict[Action, Action] = {}
    for action, (delta_row, delta_col) in _ACTION_VECTOR.items():
        origin = map_position(1, 1)
        target = map_position(1 + delta_row, 1 + delta_col)
        mapped_vector = (target[0] - origin[0], target[1] - origin[1])
        action_map[action] = _VECTOR_ACTION[mapped_vector]
    return tuple(transformed), action_map


def augment_example(example: Example) -> tuple[Example, ...]:
    """Return deterministic D4 augmentations with action labels transformed."""
    augmented: list[Example] = []
    for transform in range(8):
        state, action_map = _transform_state(example.state, transform)
        next_state = None
        if example.next_state is not None:
            next_state, _ = _transform_state(example.next_state, transform)
        augmented.append(
            Example(
                state=state,
                expert_action=action_map[example.expert_action],
                remaining_steps=example.remaining_steps,
                next_state=next_state,
                distance_bucket=example.distance_bucket,
                risk_target=example.risk_target,
            )
        )
    return tuple(augmented)


def collect_dagger_examples(
    agent: Any,
    starts: Sequence[PuzzleState | Example],
    *,
    max_steps: int = 80,
    seed: int = 42,
) -> tuple[Example, ...]:
    """Collect model-visited states and relabel them with the offline teacher."""
    rng = random.Random(seed)
    states = [start.state if isinstance(start, Example) else tuple(start) for start in starts]
    rng.shuffle(states)
    collected: list[Example] = []
    seen: set[PuzzleState] = set()
    for initial in states:
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()
        current = initial
        for step in range(max_steps):
            if current in seen or current == goal_state():
                break
            seen.add(current)
            trajectory = _teacher_trajectory(current)
            if trajectory:
                teacher_state, action, remaining_steps = trajectory[0]
                next_state = apply_action(teacher_state, action)
                collected.append(
                    Example(
                        teacher_state,
                        action,
                        remaining_steps,
                        next_state=next_state,
                        distance_bucket=min(96, remaining_steps),
                    )
                )
            trace = agent.decide(current, step=step)
            next_state = apply_action(current, trace.selected_action)
            if next_state is None:
                break
            current = next_state
    unique: dict[PuzzleState, Example] = {}
    for example in collected:
        unique.setdefault(example.state, example)
    return tuple(unique.values())


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


def generate_uniform_examples(
    count: int,
    *,
    seed: int = 42,
) -> tuple[Example, ...]:
    """Label uniformly shuffled solvable boards for hard-board training.

    Unlike ``generate_trajectory_examples``, this samples from the board
    distribution used by the interactive ``R`` reset button.  The teacher is
    still used only offline to create labels.
    """
    if count <= 0:
        return ()
    rng = random.Random(seed)
    examples: list[Example] = []
    seen: set[PuzzleState] = {goal_state()}
    attempts = 0
    max_attempts = max(100, count * 100)
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        values = list(range(1, 16)) + [-1]
        rng.shuffle(values)
        state = tuple(values)
        if state in seen or not is_solvable(state, 4, 4):
            continue
        seen.add(state)
        labeled = _label_state(state)
        if labeled is not None:
            examples.append(labeled)
    if len(examples) < count:
        raise RuntimeError(
            f"could only label {len(examples)} of {count} uniform random states"
        )
    return tuple(examples)


def generate_uniform_trajectory_examples(
    count: int,
    *,
    seed: int = 42,
) -> tuple[Example, ...]:
    """Collect labeled states along full teacher routes from random boards."""
    if count <= 0:
        return ()
    rng = random.Random(seed)
    examples: dict[PuzzleState, Example] = {}
    attempts = 0
    max_attempts = max(20, count // 8)
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        values = list(range(1, 16)) + [-1]
        rng.shuffle(values)
        state = tuple(values)
        if not is_solvable(state, 4, 4):
            continue
        trajectory = _teacher_trajectory(state)
        if not trajectory:
            continue
        for route_state, action, remaining_steps in trajectory:
            examples.setdefault(
                route_state,
                Example(
                    state=route_state,
                    expert_action=action,
                    remaining_steps=remaining_steps,
                    next_state=apply_action(route_state, action),
                    distance_bucket=min(96, remaining_steps),
                ),
            )
            if len(examples) >= count:
                break
    if len(examples) < count:
        raise RuntimeError(
            f"could only collect {len(examples)} of {count} uniform trajectory states"
        )
    return tuple(examples.values())


def generate_trajectory_examples(
    count: int,
    *,
    seed: int = 42,
    min_depth: int = 2,
    max_depth: int = 24,
) -> tuple[Example, ...]:
    """Generate unique examples from every state on offline teacher routes.

    Initial states are sampled by deterministic random walks from the goal.  A
    weighted A* route is then replayed and each pre-goal state contributes one
    supervised example.  ``max_depth`` is the random-walk difficulty budget;
    it is not a claim that the resulting state's shortest solution has that
    many moves.  ``count`` counts unique states, so overlapping routes do not
    silently duplicate training samples.
    """
    if count <= 0:
        return ()
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError("depth range must satisfy 1 <= min_depth <= max_depth")

    rng = random.Random(seed)
    examples: list[Example] = []
    seen_states: set[PuzzleState] = set()
    sampled_states: set[PuzzleState] = {goal_state()}
    attempts = 0
    max_attempts = max(80, count * 80)
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        initial_state = _random_walk_state(
            rng,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if initial_state in sampled_states:
            continue
        sampled_states.add(initial_state)

        trajectory = _teacher_trajectory(initial_state)
        if not trajectory:
            continue
        for state, action, remaining_steps in trajectory:
            if state in seen_states:
                continue
            seen_states.add(state)
            next_state = apply_action(state, action)
            examples.append(
                Example(
                    state,
                    action,
                    remaining_steps,
                    next_state=next_state,
                    distance_bucket=min(96, remaining_steps),
                )
            )
            if len(examples) >= count:
                break

    if len(examples) < count:
        raise RuntimeError(
            "could only generate "
            f"{len(examples)} of {count} unique trajectory states"
        )
    return tuple(examples)
