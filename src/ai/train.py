from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .agent import (
    AStarGuardedAgent,
    DecisionAgent,
    DecisionTrace,
    EnsembleNeuralAgent,
    NeuralBeamAgent,
    NeuralAgent,
    load_agent,
)
from .dataset import Example, generate_examples, generate_trajectory_examples
from .encoding import Action, PuzzleState, encode_state, goal_state, legal_actions
from .environment import PuzzleEnvironment
from .model import HuarongNet, load_checkpoint, save_checkpoint


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
    traces: list[DecisionTrace] = []
    for example in examples:
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()
        environment = PuzzleEnvironment(example.state, max_steps=max_steps)
        for step in range(max_steps):
            trace = agent.decide(environment.state, step=step)
            decisions += 1
            if step == 0 and trace.selected_action == example.expert_action:
                first_action_matches += 1
            if len(traces) < 3:
                traces.append(trace)
            inference_times.append(trace.inference_ms)
            fallback_decisions += int(trace.fallback_used)
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
            "average_fold_agreement": sum(agreement_values) / len(agreement_values)
            if agreement_values
            else 1.0,
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
) -> TrainingResult:
    _set_seed(seed)
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
            "train_max_depth": train_max_depth if stage == "sixth-generation" else None,
            "board": "4x4",
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
    args = parser.parse_args()
    result = run_training(**vars(args))
    print(f"checkpoint: {result.checkpoint}")
    print(f"report: {result.report}")
    print(f"metrics: {result.metrics}")


if __name__ == "__main__":
    main()
