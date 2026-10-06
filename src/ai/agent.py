from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .encoding import (
    Action,
    PuzzleState,
    apply_action,
    blank_position,
    encode_state,
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
        return DecisionTrace(
            state=tuple(state),
            blank_position=blank_position(state),
            candidates=tuple(candidates),
            selected_action=selected.action,
            confidence=float(confidence),
            inference_ms=inference_ms,
            step=step,
            hidden_summary=hidden_summary,
        )
