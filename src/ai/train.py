from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .agent import (
    AStarGuardedAgent,
    DecisionAgent,
    DecisionTrace,
    EighthGenerationNeuralAgent,
    EnsembleNeuralAgent,
    NeuralBeamAgent,
    NeuralAgent,
    load_agent,
)
from .dataset import (
    Example,
    augment_example,
    collect_dagger_examples,
    generate_examples,
    generate_uniform_examples,
    generate_uniform_trajectory_examples,
    generate_trajectory_examples,
)
from .encoding import (
    ENHANCED_FEATURE_SIZE,
    Action,
    PuzzleState,
    encode_state,
    encode_state_enhanced,
    goal_state,
    legal_actions,
)
from .environment import PuzzleEnvironment
from .model import (
    HuarongNet,
    load_checkpoint,
    save_checkpoint,
    save_ensemble_checkpoint,
)


class _TrainingProgress:
    """Dependency-free terminal progress bar for long CPU runs."""

    def __init__(self, *, total: int = 100):
        self.total = total
        self.current = 0
        self.started = perf_counter()

    def update(self, current: int, label: str) -> None:
        self.current = max(0, min(self.total, int(current)))
        width = 32
        filled = int(width * self.current / self.total)
        elapsed = perf_counter() - self.started
        eta_text = ""
        if self.current:
            eta = elapsed * (self.total - self.current) / self.current
            eta_text = f" ETA {eta / 60:.1f}m"
        line = f"\r训练进度 [{'#' * filled}{'-' * (width - filled)}] {self.current:3d}% {label}{eta_text}"
        sys.stderr.write(line[:180].ljust(180))
        sys.stderr.flush()
        if self.current >= self.total:
            sys.stderr.write("\n")


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    report: Path
    metrics: Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _split_examples(
    examples: tuple[Example, ...], seed: int, validation_ratio: float = 0.2
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_ratio))
    return tuple(shuffled[validation_count:]), tuple(shuffled[:validation_count])


def _kfold_examples(
    examples: tuple[Example, ...], *, folds: int, seed: int
) -> tuple[tuple[tuple[Example, ...], tuple[Example, ...]], ...]:
    if folds < 2:
        raise ValueError("cross-validation requires at least two folds")
    states = [example.state for example in examples]
    if len(states) != len(set(states)):
        raise ValueError("cross-validation requires unique states")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    fold_indices = [indices[index::folds] for index in range(folds)]
    result = []
    for index in range(folds):
        validation_indices = set(fold_indices[index])
        validation = tuple(examples[item] for item in fold_indices[index])
        train = tuple(examples[item] for item in indices if item not in validation_indices)
        result.append((train, validation))
    return tuple(result)


def _loader(examples: Iterable[Example], batch_size: int, shuffle: bool) -> DataLoader:
    examples = tuple(examples)
    features = torch.stack([encode_state(example.state) for example in examples])
    actions = torch.tensor([int(example.expert_action) for example in examples])
    steps = torch.tensor([-float(example.remaining_steps) for example in examples])
    return DataLoader(
        TensorDataset(features, actions, steps),
        batch_size=min(batch_size, len(examples)),
        shuffle=shuffle,
    )


def train_imitation(
    train_examples: tuple[Example, ...],
    validation_examples: tuple[Example, ...],
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    device: str = "cpu",
) -> tuple[HuarongNet, list[dict[str, float]], HuarongNet]:
    model = HuarongNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss = nn.CrossEntropyLoss()
    q_loss = nn.SmoothL1Loss()
    train_loader = _loader(train_examples, batch_size, shuffle=True)
    validation_loader = _loader(validation_examples, batch_size, shuffle=False)
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    final_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, actions, target_q in train_loader:
            features, actions, target_q = features.to(device), actions.to(device), target_q.to(device)
            optimizer.zero_grad()
            logits, q_values = model(features)
            selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = policy_loss(logits, actions) + 0.2 * q_loss(selected_q, target_q)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(features)

        model.eval()
        correct = 0
        validation_total = 0.0
        validation_count = len(validation_examples)
        with torch.inference_mode():
            for features, actions, target_q in validation_loader:
                features, actions, target_q = features.to(device), actions.to(device), target_q.to(device)
                logits, q_values = model(features)
                selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                validation_total += float(
                    (policy_loss(logits, actions) + 0.2 * q_loss(selected_q, target_q)).item()
                ) * len(features)
                correct += int((logits.argmax(dim=1) == actions).sum().item())
        validation_accuracy = correct / validation_count
        final_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(train_examples),
                "validation_loss": validation_total / len(validation_examples),
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
    best_model = HuarongNet()
    best_model.load_state_dict(best_state or model.state_dict())
    final_model = HuarongNet()
    final_model.load_state_dict(final_state or model.state_dict())
    return best_model, history, final_model


def train_cross_validation(
    examples: tuple[Example, ...],
    *,
    folds: int,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[list[HuarongNet], list[dict[str, Any]], list[list[dict[str, float]]]]:
    models: list[HuarongNet] = []
    fold_metrics: list[dict[str, Any]] = []
    histories: list[list[dict[str, float]]] = []
    for fold_index, (train_examples, validation_examples) in enumerate(
        _kfold_examples(examples, folds=folds, seed=seed), start=1
    ):
        _set_seed(seed + fold_index)
        model, history, final_model = train_imitation(
            train_examples,
            validation_examples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
        validation_evaluation, _ = evaluate_agent(model, validation_examples, device=device)
        final_evaluation, _ = evaluate_agent(final_model, validation_examples, device=device)
        selected_model = model
        selected_evaluation = validation_evaluation
        if final_evaluation["solve_rate"] > validation_evaluation["solve_rate"] or (
            final_evaluation["solve_rate"] == validation_evaluation["solve_rate"]
            and (final_evaluation["average_steps"] or float("inf"))
            < (validation_evaluation["average_steps"] or float("inf"))
        ):
            selected_model = final_model
            selected_evaluation = final_evaluation
        models.append(selected_model)
        histories.append(history)
        fold_metrics.append(
            {
                "fold": fold_index,
                "train": len(train_examples),
                "validation": len(validation_examples),
                "validation_evaluation": validation_evaluation,
                "final_evaluation": final_evaluation,
                "selected_evaluation": selected_evaluation,
            }
        )
    return models, fold_metrics, histories


def _eighth_loader(
    examples: tuple[Example, ...],
    batch_size: int,
    shuffle: bool,
    *,
    enhanced: bool = False,
) -> DataLoader:
    if not examples:
        raise ValueError("eighth-generation training requires at least one example")
    encoder = encode_state_enhanced if enhanced else lambda state: encode_state(state, include_heuristic=False)
    features = torch.stack([encoder(example.state) for example in examples])
    successor_features = torch.stack(
        [
            encoder(example.next_state or example.state)
            for example in examples
        ]
    )
    actions = torch.tensor([int(example.expert_action) for example in examples])
    remaining = torch.tensor([float(example.remaining_steps) for example in examples])
    successor_remaining = torch.clamp(remaining - 1.0, min=0.0)
    buckets = torch.tensor(
        [min(96, example.distance_bucket or example.remaining_steps) for example in examples]
    )
    legal_masks = torch.zeros((len(examples), len(Action)), dtype=torch.bool)
    risk_targets = torch.ones((len(examples), len(Action)), dtype=torch.float32)
    for index, example in enumerate(examples):
        for action in legal_actions(example.state):
            legal_masks[index, int(action)] = True
        risk_targets[index, int(example.expert_action)] = 0.0
    return DataLoader(
        TensorDataset(
            features,
            successor_features,
            actions,
            remaining,
            successor_remaining,
            buckets,
            legal_masks,
            risk_targets,
        ),
        batch_size=min(batch_size, len(examples)),
        shuffle=shuffle,
    )


def _ranking_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    selected = q_values.gather(1, actions.unsqueeze(1))
    losses: list[torch.Tensor] = []
    for index in range(len(actions)):
        alternatives = legal_masks[index].clone()
        alternatives[int(actions[index])] = False
        if alternatives.any():
            losses.append(
                torch.relu(0.25 - selected[index, 0] + q_values[index][alternatives]).mean()
            )
    if not losses:
        return q_values.sum() * 0.0
    return torch.stack(losses).mean()


def train_eighth_generation_fold(
    train_examples: tuple[Example, ...],
    validation_examples: tuple[Example, ...],
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    device: str = "cpu",
    input_size: int = 260,
    enhanced: bool = False,
    progress: _TrainingProgress | None = None,
    progress_start: int = 0,
    progress_end: int = 0,
) -> tuple[HuarongNet, list[dict[str, float]], HuarongNet]:
    model = HuarongNet(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _eighth_loader(
        train_examples, batch_size, shuffle=True, enhanced=enhanced
    )
    validation_loader = _eighth_loader(
        validation_examples, batch_size, shuffle=False, enhanced=enhanced
    )
    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        if progress is not None and progress_end > progress_start:
            progress.update(
                progress_start + (progress_end - progress_start) * (epoch - 1) // epochs,
                f"fold training epoch {epoch}/{epochs}",
            )
        model.train()
        train_total = 0.0
        for (
            features,
            successor_features,
            actions,
            remaining,
            successor_remaining,
            buckets,
            legal_masks,
            risk_targets,
        ) in train_loader:
            features = features.to(device)
            actions = actions.to(device)
            remaining = remaining.to(device)
            successor_features = successor_features.to(device)
            successor_remaining = successor_remaining.to(device)
            buckets = buckets.to(device)
            legal_masks = legal_masks.to(device)
            risk_targets = risk_targets.to(device)
            outputs = model.forward_full(features)
            successor_outputs = model.forward_full(successor_features)
            masked_policy = outputs["policy_logits"].masked_fill(~legal_masks, -1e9)
            selected_q = outputs["q_values"].gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = (
                nn.functional.cross_entropy(masked_policy, actions)
                + 0.35 * nn.functional.smooth_l1_loss(outputs["distance"].squeeze(1), remaining)
                + 0.20 * nn.functional.cross_entropy(outputs["distance_logits"], buckets)
                + 0.20 * _ranking_loss(outputs["q_values"], actions, legal_masks)
                + 0.20 * nn.functional.binary_cross_entropy_with_logits(
                    outputs["risk_logits"], risk_targets
                )
                + 0.10 * nn.functional.smooth_l1_loss(selected_q, -remaining)
                + 0.20
                * nn.functional.smooth_l1_loss(
                    successor_outputs["distance"].squeeze(1), successor_remaining
                )
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * len(features)

        model.eval()
        validation_total = 0.0
        correct = 0
        distance_correct = 0
        count = 0
        with torch.inference_mode():
            for (
                features,
                successor_features,
                actions,
                remaining,
                successor_remaining,
                buckets,
                legal_masks,
                risk_targets,
            ) in validation_loader:
                features = features.to(device)
                actions = actions.to(device)
                remaining = remaining.to(device)
                successor_features = successor_features.to(device)
                successor_remaining = successor_remaining.to(device)
                buckets = buckets.to(device)
                legal_masks = legal_masks.to(device)
                risk_targets = risk_targets.to(device)
                outputs = model.forward_full(features)
                successor_outputs = model.forward_full(successor_features)
                masked_policy = outputs["policy_logits"].masked_fill(~legal_masks, -1e9)
                selected_q = outputs["q_values"].gather(1, actions.unsqueeze(1)).squeeze(1)
                validation_loss = (
                    nn.functional.cross_entropy(masked_policy, actions)
                    + 0.35 * nn.functional.smooth_l1_loss(outputs["distance"].squeeze(1), remaining)
                    + 0.20 * nn.functional.cross_entropy(outputs["distance_logits"], buckets)
                    + 0.20 * _ranking_loss(outputs["q_values"], actions, legal_masks)
                    + 0.20 * nn.functional.binary_cross_entropy_with_logits(
                        outputs["risk_logits"], risk_targets
                    )
                    + 0.10 * nn.functional.smooth_l1_loss(selected_q, -remaining)
                    + 0.20
                    * nn.functional.smooth_l1_loss(
                        successor_outputs["distance"].squeeze(1), successor_remaining
                    )
                )
                validation_total += float(validation_loss.item()) * len(features)
                correct += int((masked_policy.argmax(dim=1) == actions).sum().item())
                distance_correct += int(
                    (outputs["distance_logits"].argmax(dim=1) == buckets).sum().item()
                )
                count += len(features)
        validation_accuracy = correct / max(1, count)
        distance_accuracy = distance_correct / max(1, count)
        final_state = copy.deepcopy(model.state_dict())
        score = 0.75 * validation_accuracy + 0.25 * distance_accuracy
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_total / max(1, len(train_examples)),
                "validation_loss": validation_total / max(1, len(validation_examples)),
                "validation_accuracy": validation_accuracy,
                "distance_accuracy": distance_accuracy,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    if progress is not None and progress_end > progress_start:
        progress.update(progress_end, "fold training complete")
    best_model = HuarongNet(input_size=input_size)
    best_model.load_state_dict(best_state or model.state_dict())
    final_model = HuarongNet(input_size=input_size)
    final_model.load_state_dict(final_state or model.state_dict())
    return best_model, history, final_model


def _trace_to_dict(trace: DecisionTrace) -> dict[str, Any]:
    return {
        "state": list(trace.state),
        "blank_position": list(trace.blank_position),
        "selected_action": trace.selected_action.name,
        "confidence": trace.confidence,
        "inference_ms": trace.inference_ms,
        "step": trace.step,
        "hidden_summary": list(trace.hidden_summary),
        "fold_agreement": trace.fold_agreement,
        "fallback_used": trace.fallback_used,
        "decision_source": trace.decision_source,
        "predicted_distance": trace.predicted_distance,
        "risk": trace.risk,
        "beam_failure": trace.beam_failure,
        "loop_detected": trace.loop_detected,
        "candidates": [
            {
                "action": candidate.action.name,
                "legal": candidate.legal,
                "probability": candidate.probability,
                "q_value": candidate.q_value,
                "heuristic_after": candidate.heuristic_after,
                "heuristic_delta": candidate.heuristic_delta,
            }
            for candidate in trace.candidates
        ],
    }


def evaluate_decider(
    agent: DecisionAgent,
    examples: tuple[Example, ...],
    *,
    max_steps: int = 120,
    device: str = "cpu",
    move_time_ms: float = 300.0,
) -> tuple[dict[str, Any], list[DecisionTrace]]:
    solved = 0
    illegal = 0
    decisions = 0
    first_action_matches = 0
    legal_decisions = 0
    fallback_decisions = 0
    agreement_values: list[float] = []
    steps_taken: list[int] = []
    inference_times: list[float] = []
    wall_times: list[float] = []
    repeated_state_visits = 0
    beam_failures = 0
    solved_30s = 0
    traces: list[DecisionTrace] = []
    for example in examples:
        episode_started = perf_counter()
        executed_steps = 0
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()
        environment = PuzzleEnvironment(example.state, max_steps=max_steps)
        seen_states: set[PuzzleState] = set()
        for step in range(max_steps):
            if environment.state in seen_states:
                repeated_state_visits += 1
            seen_states.add(environment.state)
            trace = agent.decide(environment.state, step=step)
            executed_steps = step + 1
            decisions += 1
            if step == 0 and trace.selected_action == example.expert_action:
                first_action_matches += 1
            if len(traces) < 3:
                traces.append(trace)
            inference_times.append(trace.inference_ms)
            fallback_decisions += int(trace.fallback_used)
            beam_failures += int(trace.beam_failure)
            agreement_values.append(trace.fold_agreement)
            transition = environment.step(trace.selected_action)
            if not transition.legal:
                illegal += 1
            else:
                legal_decisions += 1
            if transition.done:
                if transition.state == goal_state():
                    solved += 1
                    steps_taken.append(step + 1)
                break
        wall_time_ms = (perf_counter() - episode_started) * 1000.0 + executed_steps * move_time_ms
        wall_times.append(wall_time_ms)
        if environment.state == goal_state() and wall_time_ms <= 30_000.0:
            solved_30s += 1
    count = len(examples)
    return (
        {
            "games": count,
            "solved": solved,
            "solve_rate": solved / count if count else 0.0,
            "average_steps": sum(steps_taken) / len(steps_taken) if steps_taken else None,
            "illegal_actions": illegal,
            "illegal_action_rate": illegal / max(1, decisions),
            "action_accuracy": first_action_matches / count if count else 0.0,
            "legal_action_accuracy": legal_decisions / max(1, decisions),
            "average_inference_ms": sum(inference_times) / len(inference_times)
            if inference_times
            else 0.0,
            "fallback_rate": fallback_decisions / max(1, decisions),
            "beam_failure_rate": beam_failures / max(1, decisions),
            "repeated_state_visits": repeated_state_visits,
            "average_fold_agreement": sum(agreement_values) / len(agreement_values)
            if agreement_values
            else 1.0,
            "solve_rate_30s": solved_30s / count if count else 0.0,
            "average_wall_time_ms": sum(wall_times) / len(wall_times) if wall_times else 0.0,
            "p95_wall_time_ms": sorted(wall_times)[max(0, int(len(wall_times) * 0.95) - 1)]
            if wall_times
            else 0.0,
            "astar_average_steps": sum(example.remaining_steps for example in examples) / count
            if count
            else None,
        },
        traces,
    )


def evaluate_agent(
    model: HuarongNet,
    examples: tuple[Example, ...],
    *,
    max_steps: int = 120,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[DecisionTrace]]:
    return evaluate_decider(
        NeuralAgent(model, device=device), examples, max_steps=max_steps, device=device
    )


def evaluate_random(
    examples: tuple[Example, ...], *, max_steps: int = 120, seed: int = 1234
) -> dict[str, Any]:
    rng = random.Random(seed)
    solved = 0
    steps_taken: list[int] = []
    for example in examples:
        environment = PuzzleEnvironment(example.state, max_steps=max_steps)
        for step in range(max_steps):
            transition = environment.step(rng.choice(legal_actions(environment.state)))
            if transition.done:
                if transition.state == goal_state():
                    solved += 1
                    steps_taken.append(step + 1)
                break
    count = len(examples)
    return {
        "games": count,
        "solved": solved,
        "solve_rate": solved / count if count else 0.0,
        "average_steps": sum(steps_taken) / len(steps_taken) if steps_taken else None,
    }


def evaluate_depth_buckets(
    agent: DecisionAgent,
    *,
    games_per_bucket: int,
    seed: int,
    depths: tuple[int, ...] = (12, 16, 20, 24),
    max_steps: int = 160,
    device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Evaluate one pure decision agent on independently sampled depth buckets."""
    results: dict[str, dict[str, Any]] = {}
    for index, depth in enumerate(depths):
        examples = generate_examples(
            games_per_bucket,
            seed=seed + index,
            min_depth=depth,
            max_depth=depth,
        )
        evaluation, _ = evaluate_decider(
            agent, examples, max_steps=max_steps, device=device
        )
        results[str(depth)] = evaluation
    return results


def _legal_action_mask(state: PuzzleState) -> torch.Tensor:
    """Return a boolean mask for actions that can be applied to ``state``.

    DQN targets must only bootstrap from actions that the environment can
    execute.  Keeping this conversion in one place also makes replay entries
    independent of the feature encoding (which does not expose legal moves).
    """

    mask = torch.zeros(len(Action), dtype=torch.bool)
    for action in legal_actions(state):
        mask[int(action)] = True
    return mask


def _double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    next_legal_mask: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Compute masked Double-DQN bootstrap targets.

    The online network chooses the next action, while the target network
    evaluates it.  The action selection is masked before ``argmax`` so an
    out-of-bounds movement can never inflate a target value.
    """

    if online_q.shape != target_q.shape:
        raise ValueError("online and target Q tensors must have the same shape")
    if online_q.ndim != 2 or online_q.shape[1] != len(Action):
        raise ValueError("Q tensors must have shape (batch, action_count)")
    if next_legal_mask.shape != online_q.shape:
        raise ValueError("next-state legal-action mask must match Q tensor shape")

    legal_mask = next_legal_mask.to(dtype=torch.bool)
    if not torch.all(legal_mask.any(dim=1)):
        raise ValueError("each next state must have at least one legal action")
    masked_online_q = online_q.masked_fill(~legal_mask, float("-inf"))
    next_actions = masked_online_q.argmax(dim=1)
    next_values = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    return rewards + gamma * (1.0 - dones) * next_values


def train_dqn(
    model: HuarongNet,
    starts: tuple[Example, ...],
    *,
    episodes: int,
    seed: int,
    eval_examples: tuple[Example, ...],
    max_steps: int = 80,
    device: str = "cpu",
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    target_model = HuarongNet()
    target_model.load_state_dict(model.state_dict())
    target_model.to(device).eval()
    replay: deque[
        tuple[torch.Tensor, Action, float, torch.Tensor, torch.Tensor, bool]
    ] = deque(maxlen=20_000)
    history: list[dict[str, float]] = []
    gamma = 0.99
    epsilon = 1.0
    updates = 0
    best_state = copy.deepcopy(model.state_dict())
    best_solve_rate = -1.0
    best_steps = float("inf")
    eval_interval = max(1, episodes // 10)
    for episode in range(episodes):
        environment = PuzzleEnvironment(rng.choice(starts).state, max_steps=max_steps)
        episode_reward = 0.0
        for _ in range(max_steps):
            state = environment.state
            state_features = encode_state(state)
            legal = legal_actions(state)
            if rng.random() < epsilon:
                action = rng.choice(legal)
            else:
                with torch.inference_mode():
                    _, q_values = model(state_features.unsqueeze(0).to(device))
                action = max(legal, key=lambda candidate: float(q_values[0, int(candidate)].item()))
            transition = environment.step(action)
            next_state_features = encode_state(transition.state)
            next_legal_mask = _legal_action_mask(transition.state)
            replay.append(
                (
                    state_features,
                    action,
                    transition.reward,
                    next_state_features,
                    next_legal_mask,
                    transition.done,
                )
            )
            episode_reward += transition.reward
            if len(replay) >= 64:
                batch = rng.sample(replay, 32)
                states, actions, rewards, next_states, next_legal_masks, dones = zip(*batch)
                state_tensor = torch.stack(states).to(device)
                next_tensor = torch.stack(next_states).to(device)
                next_legal_mask_tensor = torch.stack(next_legal_masks).to(device)
                action_tensor = torch.tensor([int(action) for action in actions], device=device)
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                done_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                _, q_values = model(state_tensor)
                selected = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    online_next_q = model(next_tensor)[1]
                    target_next_q = target_model(next_tensor)[1]
                    targets = _double_dqn_targets(
                        reward_tensor,
                        done_tensor,
                        online_next_q,
                        target_next_q,
                        next_legal_mask_tensor,
                        gamma=gamma,
                    )
                loss = nn.functional.smooth_l1_loss(selected, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                updates += 1
                if updates % 100 == 0:
                    target_model.load_state_dict(model.state_dict())
            if transition.done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        if episode == 0 or (episode + 1) % max(1, episodes // 20) == 0:
            history.append({"episode": float(episode + 1), "reward": episode_reward, "epsilon": epsilon})
        if (episode + 1) % eval_interval == 0 or episode + 1 == episodes:
            checkpoint_evaluation, _ = evaluate_agent(model, eval_examples, device=device)
            current_steps = checkpoint_evaluation["average_steps"] or float("inf")
            history[-1].update(
                {
                    "eval_solve_rate": checkpoint_evaluation["solve_rate"],
                    "eval_average_steps": current_steps,
                }
            )
            if checkpoint_evaluation["solve_rate"] > best_solve_rate or (
                checkpoint_evaluation["solve_rate"] == best_solve_rate and current_steps < best_steps
            ):
                best_state = copy.deepcopy(model.state_dict())
                best_solve_rate = checkpoint_evaluation["solve_rate"]
                best_steps = current_steps
    model.load_state_dict(best_state)
    return history


def evaluate_uniform_random(
    agent: DecisionAgent,
    *,
    games: int,
    seed: int,
    max_steps: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate on the same uniform solvable distribution used by the game reset."""
    examples = generate_uniform_examples(games, seed=seed)
    evaluation, _ = evaluate_decider(
        agent, examples, max_steps=max_steps, device=device
    )
    return evaluation


def write_training_report(
    metrics: dict[str, Any],
    traces: list[DecisionTrace],
    report_path: Path,
    metrics_path: Path,
) -> None:
    metrics = dict(metrics)
    metrics["decision_traces"] = [_trace_to_dict(trace) for trace in traces]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = metrics.get("evaluation", {})
    lines = [
        "# DigitalHuarongPass Neural Solver Training Report",
        "",
        f"- Stage: `{metrics.get('stage', 'unknown')}`",
        f"- Git commit: `{metrics.get('git_sha', 'unknown')}`",
        f"- Device: `{metrics.get('device', 'cpu')}`",
        f"- Seed: `{metrics.get('seed', 'unknown')}`",
        "",
        "## Dataset And Configuration",
        "",
        "```json",
        json.dumps(metrics.get("config", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "Dataset statistics: "
        f"requested `{metrics.get('dataset', {}).get('requested', 'n/a')}`, "
        f"train `{metrics.get('dataset', {}).get('train', 'n/a')}`, "
        f"validation `{metrics.get('dataset', {}).get('validation', 'n/a')}`, "
        f"label failures `{metrics.get('dataset', {}).get('label_failures', 0)}`.",
        "",
        "## Evaluation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Games | {evaluation.get('games', 0)} |",
        f"| Solved | {evaluation.get('solved', 0)} |",
        f"| Solve rate | {evaluation.get('solve_rate', 0):.2%} |",
        f"| Average steps | {evaluation.get('average_steps', 'n/a')} |",
        f"| A* average steps | {evaluation.get('astar_average_steps', 'n/a')} |",
        f"| First-action accuracy | {evaluation.get('action_accuracy', 0):.2%} |",
        f"| Legal-action accuracy | {evaluation.get('legal_action_accuracy', 0):.2%} |",
        f"| Illegal action rate | {evaluation.get('illegal_action_rate', 0):.2%} |",
        f"| Average inference time | {evaluation.get('average_inference_ms', 0):.3f} ms |",
        f"| Average fold agreement | {evaluation.get('average_fold_agreement', 1):.2%} |",
        f"| Fallback rate | {evaluation.get('fallback_rate', 0):.2%} |",
        "",
        "## Baselines And Checkpoint Selection",
        "",
    ]
    baseline = metrics.get("baseline_evaluation")
    if baseline:
        lines.extend(
            [
                f"The DQN evaluation was compared with the supervised checkpoint: solve rate `{baseline.get('solve_rate', 0):.2%}`, average steps `{baseline.get('average_steps', 'n/a')}`.",
                f"Selection: {metrics.get('selection_reason', 'no selection reason recorded')}",
                "",
            ]
        )
    elif metrics.get("selection_reason"):
        lines.extend([f"Selection: {metrics['selection_reason']}", ""])
    random_baseline = evaluation.get("random_baseline")
    if random_baseline:
        lines.extend(
            [
                f"Random legal-action baseline: solve rate `{random_baseline.get('solve_rate', 0):.2%}`, average steps `{random_baseline.get('average_steps', 'n/a')}`.",
                "",
            ]
        )
    lines.extend(
        [
        "## Training History",
        "",
        "```json",
        json.dumps(metrics.get("history", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Decision Trace Samples",
        "",
        ]
    )
    fold_metrics = metrics.get("cross_validation", {}).get("folds", [])
    if fold_metrics:
        insert_at = lines.index("## Training History")
        fold_lines = [
            "## Cross-Validation",
            "",
            "Each state appears in exactly one validation fold. The ensemble averages all fold models at inference time.",
            "",
            "| Fold | Train states | Validation states | Validation solve rate | First-action accuracy |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for fold in fold_metrics:
            evaluation = fold.get("selected_evaluation", {})
            fold_lines.append(
                f"| {fold.get('fold', '?')} | {fold.get('train', 0)} | {fold.get('validation', 0)} | {evaluation.get('solve_rate', 0):.2%} | {evaluation.get('action_accuracy', 0):.2%} |"
            )
        fold_lines.extend([""])
        lines[insert_at:insert_at] = fold_lines
    guarded = metrics.get("guarded_evaluation")
    if guarded:
        insert_at = lines.index("## Training History")
        guarded_lines = [
            "## Guarded Evaluation",
            "",
            "The pure ensemble remains the primary neural result. The guarded result uses A* only when the ensemble confidence or fold agreement is below the recorded threshold.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Solve rate | {guarded.get('solve_rate', 0):.2%} |",
            f"| Average steps | {guarded.get('average_steps', 'n/a')} |",
            f"| Fallback rate | {guarded.get('fallback_rate', 0):.2%} |",
            "",
        ]
        lines[insert_at:insert_at] = guarded_lines
    depth_buckets = metrics.get("depth_buckets", {})
    if depth_buckets:
        insert_at = lines.index("## Training History")
        depth_lines = [
            "## Pure Neural Depth Buckets",
            "",
            "These evaluations disable the A* guard and measure the deployed neural ensemble directly.",
            "",
            "| Scramble depth | Games | Solve rate | Average steps | Illegal action rate | Fallback rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for depth, bucket in sorted(depth_buckets.items(), key=lambda item: int(item[0])):
            depth_lines.append(
                f"| {depth} | {bucket.get('games', 0)} | {bucket.get('solve_rate', 0):.2%} | "
                f"{bucket.get('average_steps', 'n/a')} | {bucket.get('illegal_action_rate', 0):.2%} | "
                f"{bucket.get('fallback_rate', 0):.2%} |"
            )
        depth_lines.extend([""])
        lines[insert_at:insert_at] = depth_lines
    uniform_evaluation = metrics.get("uniform_random_evaluation")
    if uniform_evaluation:
        insert_at = lines.index("## Training History")
        uniform_lines = [
            "## Uniform Random Board Evaluation",
            "",
            "This evaluation uses uniformly shuffled solvable boards, matching the interactive random reset distribution.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Games | {uniform_evaluation.get('games', 0)} |",
            f"| Solved | {uniform_evaluation.get('solved', 0)} |",
            f"| Solve rate | {uniform_evaluation.get('solve_rate', 0):.2%} |",
            f"| Solve rate within 30 seconds | {uniform_evaluation.get('solve_rate_30s', 0):.2%} |",
            f"| Illegal action rate | {uniform_evaluation.get('illegal_action_rate', 0):.2%} |",
            f"| Beam failure rate | {uniform_evaluation.get('beam_failure_rate', 0):.2%} |",
            f"| Repeated state visits | {uniform_evaluation.get('repeated_state_visits', 0)} |",
            "",
        ]
        lines[insert_at:insert_at] = uniform_lines
    for index, trace in enumerate(traces, start=1):
        lines.extend(
            [
                f"### Sample {index}",
                "",
                f"Selected `{trace.selected_action.name}`, confidence `{trace.confidence:.3f}`, inference `{trace.inference_ms:.3f} ms`.",
                "",
                "| Action | Legal | Probability | Q value | Heuristic delta |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for candidate in trace.candidates:
            lines.append(
                f"| {candidate.action.name} | {'yes' if candidate.legal else 'no'} | {candidate.probability:.3f} | {candidate.q_value:.3f} | {candidate.heuristic_delta if candidate.heuristic_delta is not None else 'n/a'} |"
            )
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_training(
    *,
    stage: str = "imitation",
    checkpoint: Path | None = None,
    samples: int = 2500,
    epochs: int = 30,
    episodes: int = 1000,
    eval_games: int = 100,
    folds: int = 5,
    train_max_depth: int = 1000,
    seed: int = 42,
    output_root: Path = Path("."),
    device: str = "cpu",
    video_output: Path | None = None,
    video_games: int = 4,
    dagger_episodes: int = 0,
) -> TrainingResult:
    _set_seed(seed)
    progress = _TrainingProgress() if stage == "eighth-generation" else None
    if progress is not None:
        progress.update(1, "initializing")
    output_root = Path(output_root)
    model_dir = output_root / "models"
    report_dir = output_root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = report_dir / f"training-{timestamp}.md"
    metrics_path = report_dir / f"training-{timestamp}.metrics.json"

    eval_examples = generate_examples(eval_games, seed=seed + 1000, min_depth=2, max_depth=10)
    metrics: dict[str, Any] = {
        "stage": stage,
        "git_sha": _git_sha(),
        "device": device,
        "seed": seed,
        "config": {
            "samples": None if stage == "eval-only" else samples,
            "epochs": epochs,
            "episodes": episodes,
            "eval_games": eval_games,
            "folds": folds,
            "train_max_depth": (
                train_max_depth
                if stage in ("sixth-generation", "eighth-generation")
                else None
            ),
            "board": "4x4",
            "video_games": video_games,
            "dagger_episodes": dagger_episodes,
        },
    }

    agent: DecisionAgent | None = None
    if stage == "imitation":
        examples = generate_examples(samples, seed=seed, min_depth=2, max_depth=12)
        train_examples, validation_examples = _split_examples(examples, seed)
        validation_model, history, final_model = train_imitation(
            train_examples,
            validation_examples,
            epochs=epochs,
            device=device,
        )
        validation_path = model_dir / "imitation-validation.pt"
        save_checkpoint(validation_path, validation_model, {"stage": "imitation-validation", "seed": seed})
        validation_evaluation, _ = evaluate_agent(validation_model, eval_examples, device=device)
        final_evaluation, _ = evaluate_agent(final_model, eval_examples, device=device)
        model = validation_model
        if final_evaluation["solve_rate"] > validation_evaluation["solve_rate"] or (
            final_evaluation["solve_rate"] == validation_evaluation["solve_rate"]
            and (final_evaluation["average_steps"] or float("inf"))
            < (validation_evaluation["average_steps"] or float("inf"))
        ):
            final_path = model_dir / "final.pt"
            save_checkpoint(final_path, final_model, {"stage": "imitation-final", "seed": seed})
            selected_path = model_dir / "第五代超级数字华容道之神.pt"
            shutil.copyfile(final_path, selected_path)
            model = final_model
            metrics["selection_reason"] = "Final epoch improved held-out solving over the validation-best checkpoint."
        else:
            selected_path = model_dir / "第五代超级数字华容道之神.pt"
            shutil.copyfile(validation_path, selected_path)
            metrics["selection_reason"] = "Validation-best checkpoint retained because it solved at least as many held-out states."
        checkpoint_path = model_dir / "imitation.pt"
        shutil.copyfile(selected_path, checkpoint_path)
        agent = NeuralAgent(model, device=device)
        metrics["validation_checkpoint_evaluation"] = validation_evaluation
        metrics["final_checkpoint_evaluation"] = final_evaluation
        metrics["dataset"] = {
            "requested": samples,
            "train": len(train_examples),
            "validation": len(validation_examples),
            "label_failures": 0,
        }
        metrics["supervised"] = {"history": history}
    elif stage == "dqn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for DQN stage")
        model, metadata = load_checkpoint(checkpoint, device=device)
        starts = generate_examples(samples, seed=seed, min_depth=2, max_depth=12)
        metrics["dataset"] = {"requested": samples, "train": len(starts), "label_failures": 0}
        history = train_dqn(
            model,
            starts,
            episodes=episodes,
            seed=seed,
            eval_examples=eval_examples,
            max_steps=80,
            device=device,
        )
        checkpoint_path = model_dir / "dqn.pt"
        save_checkpoint(checkpoint_path, model, {**metadata, "stage": "dqn", "seed": seed})
        baseline_model, _ = load_checkpoint(checkpoint, device=device)
        baseline_evaluation, _ = evaluate_agent(baseline_model, eval_examples, device=device)
        metrics["baseline_evaluation"] = baseline_evaluation
        agent = NeuralAgent(model, device=device)
    elif stage in ("cross-validation", "crossval"):
        examples = generate_examples(samples, seed=seed, min_depth=2, max_depth=16)
        models, fold_metrics, histories = train_cross_validation(
            examples,
            folds=folds,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        fold_names: list[str] = []
        for fold_index, fold_model in enumerate(models, start=1):
            fold_name = f"cv-fold-{fold_index}.pt"
            fold_names.append(fold_name)
            save_checkpoint(
                model_dir / fold_name,
                fold_model,
                {
                    "stage": "cross-validation-fold",
                    "seed": seed,
                    "fold": fold_index,
                    "folds": folds,
                },
            )
        ensemble = EnsembleNeuralAgent(models, device=device)
        guarded_agent = AStarGuardedAgent(ensemble)
        pure_evaluation, _ = evaluate_decider(ensemble, eval_examples, device=device)
        guarded_evaluation, _ = evaluate_decider(guarded_agent, eval_examples, device=device)
        selected_path = model_dir / "第五代超级数字华容道之神.pt"
        save_checkpoint(
            selected_path,
            models[0],
            {
                "stage": "cross-validation-ensemble",
                "seed": seed,
                "folds": folds,
                "ensemble_checkpoints": fold_names,
                "astar_guard": True,
                "guard_confidence": guarded_agent.confidence_threshold,
                "guard_agreement": guarded_agent.agreement_threshold,
            },
        )
        checkpoint_path = selected_path
        model = models[0]
        agent = ensemble
        metrics["dataset"] = {
            "requested": samples,
            "train": len(examples),
            "validation": len(examples) // folds,
            "label_failures": 0,
        }
        metrics["cross_validation"] = {"folds": fold_metrics, "histories": histories}
        metrics["guarded_evaluation"] = guarded_evaluation
        history = [{"folds": folds, "epochs": epochs}]
        metrics["selection_reason"] = (
            "Five state-disjoint fold models were averaged for the primary neural evaluation; "
            "the checkpoint metadata enables the A* guard in the interactive app."
        )
    elif stage == "sixth-generation":
        if train_max_depth < 2:
            raise ValueError("train_max_depth must be at least 2")
        examples = generate_trajectory_examples(
            samples,
            seed=seed,
            min_depth=2,
            max_depth=train_max_depth,
        )
        models, fold_metrics, histories = train_cross_validation(
            examples,
            folds=folds,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        fold_names: list[str] = []
        for fold_index, fold_model in enumerate(models, start=1):
            fold_name = f"sixth-generation-fold-{fold_index}.pt"
            fold_names.append(fold_name)
            save_checkpoint(
                model_dir / fold_name,
                fold_model,
                {
                    "stage": "sixth-generation-fold",
                    "seed": seed,
                    "fold": fold_index,
                    "folds": folds,
                    "train_min_depth": 2,
                    "train_max_depth": train_max_depth,
                    "astar_guard": False,
                    "neural_planner": "beam",
                    "beam_width": 64,
                    "beam_horizon": 80,
                },
            )
        # Preserve the strong fifth-generation shallow-policy folds as frozen
        # members of the sixth-generation ensemble.  The newly trained deep
        # folds add coverage for harder states without erasing the reliable
        # shallow behavior learned previously.
        base_models: list[HuarongNet] = []
        base_names: list[str] = []
        for fold_index in range(1, folds + 1):
            source = model_dir / f"cv-fold-{fold_index}.pt"
            if not source.exists():
                continue
            base_name = f"sixth-generation-base-fold-{fold_index}.pt"
            destination = model_dir / base_name
            shutil.copyfile(source, destination)
            base_model, _ = load_checkpoint(destination, device=device)
            base_models.append(base_model)
            base_names.append(base_name)
        ensemble_models = base_models + models
        ensemble = EnsembleNeuralAgent(ensemble_models, device=device)
        ensemble_names = base_names + fold_names
        selected_path = model_dir / "第七代飞天数字华容道享受者.pt"
        save_checkpoint(
            selected_path,
            models[0],
            {
                "stage": "sixth-generation",
                "seed": seed,
                "folds": folds,
                "train_min_depth": 2,
                "train_max_depth": train_max_depth,
                "ensemble_checkpoints": ensemble_names,
                "astar_guard": False,
                "offline_teacher": "weighted A*",
                "display_name": "第七代飞天数字华容道享受者",
                "neural_planner": "beam",
                "beam_width": 64,
                "beam_horizon": 80,
            },
        )
        checkpoint_path = selected_path
        agent = NeuralBeamAgent(
            ensemble,
            beam_width=64,
            horizon=80,
        )
        metrics["dataset"] = {
            "requested": samples,
            "train": len(examples),
            "validation": len(examples) // folds,
            "label_failures": 0,
            "min_depth": 2,
            "max_depth": train_max_depth,
        }
        metrics["cross_validation"] = {"folds": fold_metrics, "histories": histories}
        metrics["depth_buckets"] = evaluate_depth_buckets(
            agent,
            games_per_bucket=max(1, min(eval_games, 100)),
            seed=seed + 1000,
            device=device,
        )
        history = [{"folds": folds, "epochs": epochs}]
        metrics["selection_reason"] = (
            "Selected a pure neural ensemble combining frozen fifth-generation folds with five new folds "
            "trained on deeper offline teacher trajectories; the checkpoint explicitly disables the A* runtime guard."
        )
    elif stage == "eighth-generation":
        if train_max_depth < 2:
            raise ValueError("train_max_depth must be at least 2")
        trajectory_count = max(1, samples // 3)
        raw_examples = generate_trajectory_examples(
            trajectory_count,
            seed=seed,
            min_depth=2,
            max_depth=train_max_depth,
        )
        if progress is not None:
            progress.update(8, "trajectory labels ready")
        uniform_examples = generate_uniform_trajectory_examples(
            samples - len(raw_examples), seed=seed + 5000
        )
        if progress is not None:
            progress.update(12, "uniform random trajectory labels ready")
        combined_training: dict[PuzzleState, Example] = {
            example.state: example for example in raw_examples
        }
        combined_training.update({example.state: example for example in uniform_examples})
        raw_examples = tuple(combined_training.values())

        def train_folds(
            fold_examples: tuple[Example, ...],
            fold_epochs: int,
            *,
            progress_start: int = 12,
            progress_end: int = 70,
        ) -> tuple[list[HuarongNet], list[dict[str, Any]], list[list[dict[str, float]]]]:
            trained_models: list[HuarongNet] = []
            trained_metrics: list[dict[str, Any]] = []
            trained_histories: list[list[dict[str, float]]] = []
            for fold_index, (fold_train, fold_validation) in enumerate(
                _kfold_examples(fold_examples, folds=folds, seed=seed), start=1
            ):
                augmented_train = tuple(
                    augmented
                    for example in fold_train
                    for augmented in augment_example(example)
                )
                _set_seed(seed + fold_index)
                best_model, fold_history, final_model = train_eighth_generation_fold(
                    augmented_train,
                    fold_validation,
                    epochs=fold_epochs,
                    input_size=ENHANCED_FEATURE_SIZE,
                    enhanced=True,
                    device=device,
                    progress=progress,
                    progress_start=progress_start
                    + int((progress_end - progress_start) * (fold_index - 1) / folds),
                    progress_end=progress_start
                    + int((progress_end - progress_start) * fold_index / folds),
                )
                best_eval, _ = evaluate_agent(best_model, fold_validation, device=device)
                final_eval, _ = evaluate_agent(final_model, fold_validation, device=device)
                selected = best_model
                selected_eval = best_eval
                if final_eval["solve_rate"] > best_eval["solve_rate"] or (
                    final_eval["solve_rate"] == best_eval["solve_rate"]
                    and (final_eval["average_steps"] or float("inf"))
                    < (best_eval["average_steps"] or float("inf"))
                ):
                    selected = final_model
                    selected_eval = final_eval
                trained_models.append(selected)
                trained_histories.append(fold_history)
                trained_metrics.append(
                    {
                        "fold": fold_index,
                        "train": len(fold_train),
                        "augmented_train": len(augmented_train),
                        "validation": len(fold_validation),
                        "validation_evaluation": best_eval,
                        "final_evaluation": final_eval,
                        "selected_evaluation": selected_eval,
                    }
                )
            return trained_models, trained_metrics, trained_histories

        models, fold_metrics, histories = train_folds(raw_examples, epochs)
        initial_agent = EighthGenerationNeuralAgent(models, device=device)
        dagger_examples: tuple[Example, ...] = ()
        if dagger_episodes > 0:
            if progress is not None:
                progress.update(72, "collecting hard-board DAgger states")
            dagger_examples = collect_dagger_examples(
                initial_agent,
                uniform_examples[: max(1, min(dagger_episodes, len(uniform_examples)))],
                max_steps=96,
                seed=seed + 7000,
            )
            combined: dict[PuzzleState, Example] = {example.state: example for example in raw_examples}
            combined.update({example.state: example for example in dagger_examples})
            models, fold_metrics, histories = train_folds(
                tuple(combined.values()),
                max(1, epochs // 2),
                progress_start=72,
                progress_end=88,
            )
        if progress is not None:
            progress.update(88, "cross-validation and DAgger complete")
        checkpoint_path = model_dir / "第八代数字华容道高速王.pt"
        save_ensemble_checkpoint(
            checkpoint_path,
            models,
            {
                "stage": "eighth-generation",
                "model_name": "第八代数字华容道高速王",
                "display_name": "第八代数字华容道高速王",
                "seed": seed,
                "folds": folds,
                "train_min_depth": 2,
                "train_max_depth": train_max_depth,
                "offline_teacher": "weighted A*",
                "astar_guard": False,
                "neural_planner": "pure-neural-beam",
                "state_representation": "enhanced-v3",
                "successor_value_training": True,
                "beam_width": 32,
                "beam_horizon": 96,
                "dagger_examples": len(dagger_examples),
            },
        )
        agent = EighthGenerationNeuralAgent(
            models,
            device=device,
            beam_width=32,
            horizon=96,
        )
        metrics["dataset"] = {
            "requested": samples,
            "train": len(raw_examples),
            "validation": len(raw_examples) // folds,
            "label_failures": 0,
            "min_depth": 2,
            "max_depth": train_max_depth,
            "trajectory_examples": len(raw_examples) - len(uniform_examples),
            "uniform_random_examples": len(uniform_examples),
            "dagger_examples": len(dagger_examples),
        }
        metrics["cross_validation"] = {"folds": fold_metrics, "histories": histories}
        metrics["depth_buckets"] = evaluate_depth_buckets(
            agent,
            games_per_bucket=max(1, min(eval_games, 2)),
            seed=seed + 1000,
            depths=(12, 16, 20, 24),
            max_steps=100,
            device=device,
        )
        metrics["uniform_random_evaluation"] = evaluate_uniform_random(
            agent,
            games=eval_games,
            seed=seed + 5000,
            max_steps=100,
            device=device,
        )
        if progress is not None:
            progress.update(96, "uniform random evaluation complete")
        history = [row for fold_history in histories for row in fold_history]
        metrics["selection_reason"] = (
            "Five state-disjoint multi-task fold models were embedded into one checkpoint; "
            "runtime uses the pure neural beam without A* or heuristic scoring."
        )
    elif stage == "eval-only":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for eval-only stage")
        agent, metadata = load_agent(checkpoint, device=device, use_guard=False)
        metrics["dataset"] = {"requested": eval_games, "evaluation": eval_games, "label_failures": 0}
        history = [{"loaded_checkpoint": str(checkpoint), "stage": metadata.get("stage", "unknown")}]
        checkpoint_path = Path(checkpoint)
        if metadata.get("stage") == "sixth-generation":
            metrics["depth_buckets"] = evaluate_depth_buckets(
                agent,
                games_per_bucket=max(1, min(eval_games, 100)),
                seed=seed + 1000,
                device=device,
            )
        if metadata.get("astar_guard"):
            guarded_agent = AStarGuardedAgent(
                agent,
                confidence_threshold=float(metadata.get("guard_confidence", 0.75)),
                agreement_threshold=float(metadata.get("guard_agreement", 0.6)),
            )
            metrics["guarded_evaluation"], _ = evaluate_decider(
                guarded_agent, eval_examples, device=device
            )
    else:
        raise ValueError(f"unknown training stage: {stage}")

    if agent is None:
        raise RuntimeError("training stage did not produce a decision agent")
    evaluation, traces = evaluate_decider(agent, eval_examples, device=device)
    evaluation["random_baseline"] = evaluate_random(eval_examples, seed=seed + 2000)
    metrics["evaluation"] = evaluation
    metrics["history"] = history
    if stage not in ("imitation", "dqn"):
        selected_path = checkpoint_path
    if stage == "dqn":
        baseline = metrics["baseline_evaluation"]
        dqn_is_better = (
            evaluation["solve_rate"] > baseline["solve_rate"]
            or (
                evaluation["solve_rate"] == baseline["solve_rate"]
                and (evaluation["average_steps"] or float("inf"))
                < (baseline["average_steps"] or float("inf"))
            )
        )
        selected_path = model_dir / "第五代超级数字华容道之神.pt"
        if dqn_is_better:
            shutil.copyfile(checkpoint_path, selected_path)
            metrics["selection_reason"] = "DQN improved held-out solve rate or average steps."
        else:
            shutil.copyfile(checkpoint, selected_path)
            metrics["selection_reason"] = "Supervised checkpoint retained because DQN did not improve evaluation."
    metrics["selected_checkpoint"] = str(selected_path)
    if stage == "eighth-generation":
        from .training_video import (
            encode_training_video,
            render_comparison_frames,
            render_training_metrics,
        )

        metrics_png = report_dir / "第八代数字华容道高速王-训练指标.png"
        render_training_metrics(metrics, metrics_png)
        video_path = Path(video_output) if video_output is not None else report_dir / "第八代数字华容道高速王-训练过程.mp4"
        if video_output is not None and not video_path.is_absolute():
            video_path = output_root / video_path
        frame_dir = report_dir / f".第八代数字华容道高速王-frames-{timestamp}"
        early_agent = EighthGenerationNeuralAgent(
            [HuarongNet()], device=device, beam_width=8, horizon=24
        )
        video_examples = generate_examples(
            max(1, video_games),
            seed=seed + 3000,
            min_depth=16,
            max_depth=24,
        )
        video_states = [example.state for example in video_examples]
        frame_count = render_comparison_frames(
            early_agent,
            agent,
            video_states,
            frame_dir,
            fps=5,
            max_steps=32,
        )
        encoded = encode_training_video(frame_dir, video_path, fps=5)
        metrics["video"] = {
            "output": str(video_path),
            "frames": frame_count,
            "encoded": encoded,
            "metrics_png": str(metrics_png),
            "early_checkpoint": "untrained HuarongNet",
            "final_checkpoint": str(selected_path),
        }
        if encoded:
            shutil.rmtree(frame_dir, ignore_errors=True)
        if progress is not None:
            progress.update(100, "video and artifacts written")
    write_training_report(metrics, traces, report_path, metrics_path)
    return TrainingResult(selected_path, report_path, metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the neural solver")
    parser.add_argument(
        "--stage",
        choices=(
            "imitation",
            "dqn",
            "cross-validation",
            "crossval",
            "sixth-generation",
            "eighth-generation",
            "eval-only",
        ),
        default="imitation",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--train-max-depth",
        type=int,
        default=1000,
        help="maximum random-walk depth used for sixth-generation teacher data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-games", type=int, default=4)
    parser.add_argument("--dagger-episodes", type=int, default=0)
    args = parser.parse_args()
    result = run_training(**vars(args))
    print(f"checkpoint: {result.checkpoint}")
    print(f"report: {result.report}")
    print(f"metrics: {result.metrics}")


if __name__ == "__main__":
    main()
