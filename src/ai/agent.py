from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from time import perf_counter
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import torch

from board import Board
from board.board import END

from .encoding import (
    ACTION_DELTAS,
    Action,
    PuzzleState,
    apply_action,
    blank_position,
    encode_state,
    goal_state,
    heuristic_score,
    legal_actions,
)
from .model import HuarongNet


@dataclass(frozen=True)
class ActionScore:
    action: Action
    legal: bool
    probability: float
    q_value: float
    heuristic_after: int | None
    heuristic_delta: int | None
    combined_score: float


@dataclass(frozen=True)
class DecisionTrace:
    state: PuzzleState
    blank_position: tuple[int, int]
    candidates: tuple[ActionScore, ...]
    selected_action: Action
    confidence: float
    inference_ms: float
    step: int
    hidden_summary: tuple[float, ...]
    fold_agreement: float = 1.0
    fallback_used: bool = False
    decision_source: str = "neural"


class DecisionAgent(Protocol):
    def decide(self, state: PuzzleState, *, step: int = 0) -> DecisionTrace:
        ...


_INVERSE_ACTION = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}


def _anti_loop_trace(
    trace: DecisionTrace,
    state: PuzzleState,
    recent_states: Sequence[PuzzleState],
    previous_action: Action | None,
    visit_counts: Mapping[PuzzleState, int],
) -> DecisionTrace:
    """Avoid short neural oscillations without consulting a search solver."""
    legal_candidates = [candidate for candidate in trace.candidates if candidate.legal]
    if not legal_candidates:
        return trace

    # Never trade away a move that reaches the goal immediately.
    if trace.selected_action in {
        candidate.action
        for candidate in legal_candidates
        if candidate.heuristic_after == 0
    }:
        return trace

    recent = set(recent_states)
    safe_candidates = []
    for candidate in legal_candidates:
        next_state = apply_action(state, candidate.action)
        if next_state is None:
            continue
        if previous_action is not None and candidate.action == _INVERSE_ACTION[previous_action]:
            continue
        if next_state in recent or visit_counts.get(next_state, 0) > 0:
            continue
        safe_candidates.append(candidate)

    # If every action is blocked by short-term history, relax the history
    # constraint and choose the least visited successor while retaining the
    # immediate reverse prohibition whenever another action exists.
    if not safe_candidates:
        fallback_candidates = [
            candidate
            for candidate in legal_candidates
            if previous_action is None or candidate.action != _INVERSE_ACTION[previous_action]
        ] or legal_candidates
        safe_candidates = [
            min(
                fallback_candidates,
                key=lambda candidate: (
                    visit_counts.get(apply_action(state, candidate.action), 0),
                    -(candidate.heuristic_delta or 0),
                    -candidate.combined_score,
                ),
            )
        ]

    selected = max(safe_candidates, key=lambda candidate: candidate.combined_score)
    if selected.action == trace.selected_action:
        return trace
    return DecisionTrace(
        state=trace.state,
        blank_position=trace.blank_position,
        candidates=trace.candidates,
        selected_action=selected.action,
        confidence=selected.probability,
        inference_ms=trace.inference_ms,
        step=trace.step,
        hidden_summary=trace.hidden_summary,
        fold_agreement=trace.fold_agreement,
        fallback_used=False,
        decision_source="neural-anti-loop",
    )


class NeuralAgent:
    def __init__(
        self,
        model: HuarongNet,
        *,
        device: str | torch.device = "cpu",
        model_name: str = "untrained",
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.model_name = model_name
        self._recent_states: deque[PuzzleState] = deque(maxlen=16)
        self._previous_action: Action | None = None
        self._visit_counts: dict[PuzzleState, int] = {}

    def reset(self) -> None:
        self._recent_states.clear()
        self._previous_action = None
        self._visit_counts.clear()

    def decide(self, state: PuzzleState, *, step: int = 0) -> DecisionTrace:
        features = encode_state(state).unsqueeze(0).to(self.device)
        started = perf_counter()
        with torch.inference_mode():
            policy_logits, q_values = self.model(features)
            hidden = self.model.hidden(features)
            probabilities = torch.softmax(policy_logits[0], dim=0).cpu().tolist()
            q_scores = q_values[0].cpu().tolist()
            hidden_values = hidden[0].cpu().tolist()
        inference_ms = (perf_counter() - started) * 1000.0

        current_heuristic = heuristic_score(state)
        legal = set(legal_actions(state))
        legal_q = [q_scores[int(action)] for action in legal]
        q_min = min(legal_q) if legal_q else 0.0
        q_span = max(legal_q) - q_min if legal_q else 1.0
        candidates: list[ActionScore] = []
        for action in Action:
            next_state = apply_action(state, action)
            is_legal = action in legal
            after = heuristic_score(next_state) if next_state is not None else None
            delta = current_heuristic - after if after is not None else None
            normalized_q = (q_scores[int(action)] - q_min) / (q_span or 1.0)
            combined = (
                0.65 * probabilities[int(action)] + 0.35 * normalized_q
                if is_legal
                else float("-inf")
            )
            candidates.append(
                ActionScore(
                    action=action,
                    legal=is_legal,
                    probability=float(probabilities[int(action)]),
                    q_value=float(q_scores[int(action)]),
                    heuristic_after=after,
                    heuristic_delta=delta,
                    combined_score=float(combined),
                )
            )

        winning = [
            candidate
            for candidate in candidates
            if candidate.legal and candidate.heuristic_after == 0
        ]
        selected = winning[0] if winning else max(
            (candidate for candidate in candidates if candidate.legal),
            key=lambda candidate: candidate.combined_score,
        )
        confidence = selected.probability
        hidden_summary = tuple(float(value) for value in hidden_values[:8])
        trace = DecisionTrace(
            state=tuple(state),
            blank_position=blank_position(state),
            candidates=tuple(candidates),
            selected_action=selected.action,
            confidence=float(confidence),
            inference_ms=inference_ms,
            step=step,
            hidden_summary=hidden_summary,
        )
        trace = _anti_loop_trace(
            trace,
            state,
            self._recent_states,
            self._previous_action,
            self._visit_counts,
        )
        self._recent_states.append(tuple(state))
        self._visit_counts[tuple(state)] = self._visit_counts.get(tuple(state), 0) + 1
        self._previous_action = trace.selected_action
        return trace


class EnsembleNeuralAgent:
    """Average independently trained fold models before selecting an action."""

    def __init__(
        self,
        models: Sequence[HuarongNet],
        *,
        device: str | torch.device = "cpu",
        model_name: str = "cross-validation ensemble",
    ):
        if not models:
            raise ValueError("an ensemble requires at least one model")
        self.models = tuple(model.to(device).eval() for model in models)
        self.device = torch.device(device)
        self.model_name = model_name
        self._recent_states: deque[PuzzleState] = deque(maxlen=16)
        self._previous_action: Action | None = None
        self._visit_counts: dict[PuzzleState, int] = {}

    def reset(self) -> None:
        self._recent_states.clear()
        self._previous_action = None
        self._visit_counts.clear()

    def decide(self, state: PuzzleState, *, step: int = 0) -> DecisionTrace:
        features = encode_state(state).unsqueeze(0).to(self.device)
        started = perf_counter()
        policy_probabilities: list[list[float]] = []
        q_values: list[list[float]] = []
        hidden_values: list[list[float]] = []
        with torch.inference_mode():
            for model in self.models:
                logits, q = model(features)
                policy_probabilities.append(torch.softmax(logits[0], dim=0).cpu().tolist())
                q_values.append(q[0].cpu().tolist())
                hidden_values.append(model.hidden(features)[0].cpu().tolist())
        probabilities = [
            sum(values[index] for values in policy_probabilities) / len(policy_probabilities)
            for index in range(len(Action))
        ]
        q_scores = [
            sum(values[index] for values in q_values) / len(q_values)
            for index in range(len(Action))
        ]
        fold_actions = [
            max(legal_actions(state), key=lambda action: values[int(action)])
            for values in policy_probabilities
        ]
        legal = set(legal_actions(state))
        agreement_action = max(set(fold_actions), key=fold_actions.count)
        fold_agreement = fold_actions.count(agreement_action) / len(fold_actions)
        hidden = [
            sum(values[index] for values in hidden_values) / len(hidden_values)
            for index in range(len(hidden_values[0]))
        ]
        inference_ms = (perf_counter() - started) * 1000.0
        trace = _build_trace(
            state,
            probabilities,
            q_scores,
            hidden,
            legal,
            inference_ms,
            step,
            fold_agreement=fold_agreement,
        )
        trace = _anti_loop_trace(
            trace,
            state,
            self._recent_states,
            self._previous_action,
            self._visit_counts,
        )
        self._recent_states.append(tuple(state))
        self._visit_counts[tuple(state)] = self._visit_counts.get(tuple(state), 0) + 1
        self._previous_action = trace.selected_action
        return trace


class NeuralBeamAgent:
    """Use the ensemble itself for bounded look-ahead planning.

    This is a neural planner rather than a search-solver fallback: every branch
    is scored by the ensemble policy, and no external route generator is used.
    A plan is cached for the current episode and recomputed if the board moves
    away from the expected next state.
    """

    def __init__(
        self,
        base_agent: EnsembleNeuralAgent,
        *,
        beam_width: int = 64,
        horizon: int = 80,
    ):
        if beam_width < 1 or horizon < 1:
            raise ValueError("beam_width and horizon must be positive")
        self.base_agent = base_agent
        self.models = base_agent.models
        self.device = base_agent.device
        self.model_name = "neural beam ensemble"
        self.beam_width = beam_width
        self.horizon = horizon
        self._planned_state: PuzzleState | None = None
        self._planned_actions: deque[Action] = deque()

    def reset(self) -> None:
        self._planned_state = None
        self._planned_actions.clear()
        self.base_agent.reset()

    def _predict_policy(self, state: PuzzleState) -> list[float]:
        return self._predict_policies((state,))[0]

    def _predict_policies(self, states: Sequence[PuzzleState]) -> list[list[float]]:
        """Batch ensemble inference for all live beam states."""
        if not states:
            return []
        features = torch.stack([encode_state(state) for state in states]).to(self.device)
        probabilities: list[list[float]] = []
        with torch.inference_mode():
            for model in self.models:
                logits, _ = model(features)
                probabilities.append(torch.softmax(logits, dim=1).cpu().tolist())
        return [
            [
                sum(model_values[state_index][action_index] for model_values in probabilities)
                / len(probabilities)
                for action_index in range(len(Action))
            ]
            for state_index in range(len(states))
        ]

    def _plan(self, start: PuzzleState) -> tuple[Action, ...] | None:
        beam: list[tuple[PuzzleState, tuple[Action, ...], float, frozenset[PuzzleState]]] = [
            (start, (), 0.0, frozenset({start}))
        ]
        for _ in range(self.horizon):
            candidates: list[
                tuple[PuzzleState, tuple[Action, ...], float, frozenset[PuzzleState]]
            ] = []
            policies = self._predict_policies([state for state, _, _, _ in beam])
            for (state, actions, score, seen), probabilities in zip(beam, policies):
                if state == goal_state():
                    return actions
                for action in legal_actions(state):
                    next_state = apply_action(state, action)
                    if next_state is None or next_state in seen:
                        continue
                    candidates.append(
                        (
                            next_state,
                            actions + (action,),
                            score + math.log(max(probabilities[int(action)], 1e-9)),
                            seen | {next_state},
                        )
                    )
            if not candidates:
                return None
            candidates.sort(key=lambda candidate: candidate[2], reverse=True)
            beam = candidates[: self.beam_width]
            for state, actions, _, _ in beam:
                if state == goal_state():
                    return actions
        return None

    def decide(self, state: PuzzleState, *, step: int = 0) -> DecisionTrace:
        state = tuple(state)
        expected = self._planned_state
        if expected != state or not self._planned_actions:
            plan = self._plan(state)
            self._planned_state = state
            self._planned_actions = deque(plan or ())

        trace = self.base_agent.decide(state, step=step)
        if not self._planned_actions:
            return trace

        selected_action = self._planned_actions.popleft()
        next_state = apply_action(state, selected_action)
        self._planned_state = next_state
        candidate = next(
            candidate
            for candidate in trace.candidates
            if candidate.action == selected_action
        )
        return DecisionTrace(
            state=trace.state,
            blank_position=trace.blank_position,
            candidates=trace.candidates,
            selected_action=selected_action,
            confidence=candidate.probability,
            inference_ms=trace.inference_ms,
            step=trace.step,
            hidden_summary=trace.hidden_summary,
            fold_agreement=trace.fold_agreement,
            fallback_used=False,
            decision_source="neural-beam",
        )


def _build_trace(
    state: PuzzleState,
    probabilities: Sequence[float],
    q_scores: Sequence[float],
    hidden_values: Sequence[float],
    legal: set[Action],
    inference_ms: float,
    step: int,
    *,
    fold_agreement: float = 1.0,
) -> DecisionTrace:
    current_heuristic = heuristic_score(state)
    legal_q = [q_scores[int(action)] for action in legal]
    q_min = min(legal_q) if legal_q else 0.0
    q_span = max(legal_q) - q_min if legal_q else 1.0
    candidates: list[ActionScore] = []
    for action in Action:
        next_state = apply_action(state, action)
        is_legal = action in legal
        after = heuristic_score(next_state) if next_state is not None else None
        delta = current_heuristic - after if after is not None else None
        normalized_q = (q_scores[int(action)] - q_min) / (q_span or 1.0)
        combined = (
            0.65 * probabilities[int(action)] + 0.35 * normalized_q
            if is_legal
            else float("-inf")
        )
        candidates.append(
            ActionScore(
                action=action,
                legal=is_legal,
                probability=float(probabilities[int(action)]),
                q_value=float(q_scores[int(action)]),
                heuristic_after=after,
                heuristic_delta=delta,
                combined_score=float(combined),
            )
        )
    winning = [
        candidate
        for candidate in candidates
        if candidate.legal and candidate.heuristic_after == 0
    ]
    selected = winning[0] if winning else max(
        (candidate for candidate in candidates if candidate.legal),
        key=lambda candidate: candidate.combined_score,
    )
    return DecisionTrace(
        state=tuple(state),
        blank_position=blank_position(state),
        candidates=tuple(candidates),
        selected_action=selected.action,
        confidence=float(selected.probability),
        inference_ms=float(inference_ms),
        step=step,
        hidden_summary=tuple(float(value) for value in hidden_values[:8]),
        fold_agreement=float(fold_agreement),
    )


class AStarGuardedAgent:
    """Use the neural proposal when confident, otherwise ask the A* teacher."""

    def __init__(
        self,
        base_agent: DecisionAgent,
        *,
        confidence_threshold: float = 0.75,
        agreement_threshold: float = 0.6,
    ):
        self.base_agent = base_agent
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        self._seen_states: set[PuzzleState] = set()

    def reset(self) -> None:
        """Start loop detection from a fresh game state."""
        self._seen_states.clear()
        base_reset = getattr(self.base_agent, "reset", None)
        if callable(base_reset):
            base_reset()

    def decide(self, state: PuzzleState, *, step: int = 0) -> DecisionTrace:
        trace = self.base_agent.decide(state, step=step)
        state_repeated = state in self._seen_states
        self._seen_states.add(state)
        proposed_state = apply_action(state, trace.selected_action)
        would_repeat = proposed_state in self._seen_states if proposed_state is not None else False
        if (
            trace.confidence >= self.confidence_threshold
            and trace.fold_agreement >= self.agreement_threshold
            and not state_repeated
            and not would_repeat
        ):
            return trace
        action = astar_action(state)
        if action is None or action == trace.selected_action:
            return trace
        loop_break = state_repeated or would_repeat
        return DecisionTrace(
            state=trace.state,
            blank_position=trace.blank_position,
            candidates=trace.candidates,
            selected_action=action,
            confidence=trace.confidence,
            inference_ms=trace.inference_ms,
            step=trace.step,
            hidden_summary=trace.hidden_summary,
            fold_agreement=trace.fold_agreement,
            fallback_used=True,
            decision_source="astar-loop-break" if loop_break else "astar-fallback",
        )


def astar_action(state: PuzzleState) -> Action | None:
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
    return direction_to_action.get(direction)


def load_agent(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_guard: bool = True,
) -> tuple[DecisionAgent, dict[str, object]]:
    """Load a single checkpoint or the ensemble metadata embedded in one."""
    from .model import load_checkpoint

    checkpoint = Path(checkpoint)
    model, metadata = load_checkpoint(checkpoint, device=device)
    ensemble_names = metadata.get("ensemble_checkpoints")
    if isinstance(ensemble_names, list) and ensemble_names:
        models = []
        for name in ensemble_names:
            fold_path = checkpoint.parent / str(name)
            if fold_path.resolve() == checkpoint.resolve():
                continue
            fold_model, _ = load_checkpoint(fold_path, device=device)
            models.append(fold_model)
        if not models:
            models = [model]
        agent: DecisionAgent = EnsembleNeuralAgent(models, device=device)
        if metadata.get("neural_planner") == "beam":
            agent = NeuralBeamAgent(
                agent,
                beam_width=int(metadata.get("beam_width", 64)),
                horizon=int(metadata.get("beam_horizon", 80)),
            )
        if use_guard and metadata.get("astar_guard"):
            agent = AStarGuardedAgent(
                agent,
                confidence_threshold=float(metadata.get("guard_confidence", 0.75)),
                agreement_threshold=float(metadata.get("guard_agreement", 0.6)),
            )
        return agent, metadata
    return NeuralAgent(model, device=device), metadata
