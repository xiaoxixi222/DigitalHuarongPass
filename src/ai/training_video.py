from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pygame

from .agent import DecisionAgent, DecisionTrace
from .encoding import Action, PuzzleState, apply_action, goal_state


def _font(size: int) -> pygame.font.Font:
    return pygame.font.Font(None, size)


def _text(surface: pygame.Surface, text: str, x: int, y: int, size: int = 24, color=(35, 42, 52)) -> None:
    surface.blit(_font(size).render(text, True, color), (x, y))


def _draw_board(surface: pygame.Surface, state: PuzzleState, rect: pygame.Rect, trace: DecisionTrace | None = None) -> None:
    pygame.draw.rect(surface, (248, 249, 251), rect, border_radius=8)
    size = min(rect.width, rect.height) // 4
    start_x = rect.x + (rect.width - size * 4) // 2
    start_y = rect.y + (rect.height - size * 4) // 2
    for index, value in enumerate(state):
        row, col = divmod(index, 4)
        tile_rect = pygame.Rect(start_x + col * size, start_y + row * size, size - 3, size - 3)
        color = (232, 238, 248) if value != -1 else (245, 245, 245)
        pygame.draw.rect(surface, color, tile_rect, border_radius=5)
        pygame.draw.rect(surface, (80, 91, 108), tile_rect, 1, border_radius=5)
        if value != -1:
            label = _font(max(18, size // 3)).render(str(value), True, (35, 42, 52))
            surface.blit(label, label.get_rect(center=tile_rect.center))
    if trace is not None:
        _text(surface, f"action {trace.selected_action.name}", rect.x + 12, rect.bottom - 34, 20, (32, 112, 210))
        predicted = "?" if trace.predicted_distance is None else f"{trace.predicted_distance:.1f}"
        _text(surface, f"distance {predicted}  confidence {trace.confidence:.2f}", rect.x + 12, rect.bottom - 12, 16, (105, 115, 128))


def render_training_metrics(metrics: dict[str, Any], output_png: str | Path) -> Path:
    """Render a dependency-free training curve PNG using pygame."""
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    pygame.init()
    surface = pygame.Surface((1280, 720))
    surface.fill((255, 255, 255))
    _text(surface, "Eighth-generation training", 40, 28, 34)
    history = metrics.get("history", [])
    points = [row for row in history if isinstance(row, dict)]
    chart = pygame.Rect(70, 110, 1140, 500)
    pygame.draw.rect(surface, (248, 249, 251), chart, border_radius=8)
    pygame.draw.rect(surface, (210, 216, 224), chart, 1, border_radius=8)
    if points:
        losses = [float(row.get("train_loss", row.get("loss", 0.0))) for row in points]
        accuracies = [float(row.get("validation_accuracy", row.get("action_accuracy", 0.0))) for row in points]
        max_loss = max(losses) or 1.0
        count = max(1, len(points) - 1)
        for index in range(1, len(points)):
            x1 = chart.x + int((index - 1) / count * chart.width)
            x2 = chart.x + int(index / count * chart.width)
            y1 = chart.bottom - 40 - int(losses[index - 1] / max_loss * (chart.height - 80))
            y2 = chart.bottom - 40 - int(losses[index] / max_loss * (chart.height - 80))
            pygame.draw.line(surface, (32, 112, 210), (x1, y1), (x2, y2), 3)
            ay1 = chart.bottom - 40 - int(max(0.0, min(1.0, accuracies[index - 1])) * (chart.height - 80))
            ay2 = chart.bottom - 40 - int(max(0.0, min(1.0, accuracies[index])) * (chart.height - 80))
            pygame.draw.line(surface, (35, 150, 102), (x1, ay1), (x2, ay2), 3)
        _text(surface, "blue: train loss", chart.x + 20, chart.y + 18, 18, (32, 112, 210))
        _text(surface, "green: validation accuracy", chart.x + 220, chart.y + 18, 18, (35, 150, 102))
    evaluation = metrics.get("evaluation", {})
    _text(surface, f"30s solve rate: {float(evaluation.get('solve_rate_30s', evaluation.get('solve_rate', 0.0))):.1%}", 40, 650, 22)
    pygame.image.save(surface, output)
    pygame.quit()
    return output


def _advance(state: PuzzleState, trace: DecisionTrace | None) -> PuzzleState:
    if trace is None:
        return state
    next_state = apply_action(state, trace.selected_action)
    return state if next_state is None else next_state


def render_comparison_frames(
    early_agent: DecisionAgent,
    final_agent: DecisionAgent,
    states: Sequence[PuzzleState],
    frame_dir: str | Path,
    fps: int = 5,
    *,
    max_steps: int = 80,
) -> int:
    """Render fixed-board early-versus-final inference frames."""
    del fps
    directory = Path(frame_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pygame.init()
    frame_index = 0
    for board_index, initial in enumerate(states):
        early_reset = getattr(early_agent, "reset", None)
        final_reset = getattr(final_agent, "reset", None)
        if callable(early_reset):
            early_reset()
        if callable(final_reset):
            final_reset()
        early_state = tuple(initial)
        final_state = tuple(initial)
        for step in range(max_steps):
            early_trace = None if early_state == goal_state() else early_agent.decide(early_state, step=step)
            final_trace = None if final_state == goal_state() else final_agent.decide(final_state, step=step)
            surface = pygame.Surface((1280, 720))
            surface.fill((255, 255, 255))
            _text(surface, "Eighth-generation training comparison", 40, 24, 34)
            _text(surface, f"board {board_index + 1}  step {step + 1}", 40, 66, 22, (105, 115, 128))
            _text(surface, "early checkpoint", 70, 112, 24, (205, 78, 78))
            _text(surface, "final ensemble", 680, 112, 24, (35, 150, 102))
            _draw_board(surface, early_state, pygame.Rect(40, 150, 560, 500), early_trace)
            _draw_board(surface, final_state, pygame.Rect(640, 150, 560, 500), final_trace)
            path = directory / f"frame-{frame_index:06d}.png"
            pygame.image.save(surface, path)
            frame_index += 1
            early_state = _advance(early_state, early_trace)
            final_state = _advance(final_state, final_trace)
            if early_state == goal_state() and final_state == goal_state():
                break
    pygame.quit()
    return frame_index


def encode_training_video(frame_dir: str | Path, output_mp4: str | Path, fps: int = 5) -> bool:
    """Encode frames with ffmpeg; return False when the optional encoder is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    output = Path(output_mp4)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(Path(frame_dir) / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and output.exists()

