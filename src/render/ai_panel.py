from __future__ import annotations

from collections.abc import Sequence

import pygame

from ai.agent import DecisionTrace
from ai.encoding import ACTION_DELTAS


PANEL_BACKGROUND = (244, 246, 249)
PANEL_BORDER = (210, 216, 224)
TEXT_COLOR = (35, 42, 52)
MUTED_COLOR = (105, 115, 128)
ACCENT_COLOR = (32, 112, 210)
GOOD_COLOR = (35, 150, 102)
BAD_COLOR = (205, 78, 78)
BAR_BACKGROUND = (220, 226, 234)


def _label(surface: pygame.Surface, text: str, position: tuple[int, int], size: int = 20, color=TEXT_COLOR) -> None:
    font = pygame.font.Font(None, size)
    surface.blit(font.render(text, True, color), position)


def _bar(surface: pygame.Surface, rect: pygame.Rect, fraction: float, color) -> None:
    pygame.draw.rect(surface, BAR_BACKGROUND, rect, border_radius=3)
    width = max(0, min(rect.width, int(rect.width * fraction)))
    if width:
        pygame.draw.rect(surface, color, pygame.Rect(rect.x, rect.y, width, rect.height), border_radius=3)


def draw_ai_panel(
    screen: pygame.Surface,
    area: pygame.Rect,
    trace: DecisionTrace | None,
    *,
    model_name: str = "neural solver",
    paused: bool = False,
    error: str | None = None,
    recent_actions: Sequence[str] = (),
) -> None:
    pygame.draw.rect(screen, PANEL_BACKGROUND, area)
    pygame.draw.line(screen, PANEL_BORDER, area.topleft, area.bottomleft, 2)
    x = area.x + 18
    width = area.width - 36
    _label(screen, "AI DECISION TRACE", (x, area.y + 16), 25)
    _label(screen, f"model: {model_name}", (x, area.y + 48), 18, MUTED_COLOR)
    status = "PAUSED / STEP" if paused else "RUNNING"
    _label(screen, f"status: {status}", (x, area.y + 70), 18, ACCENT_COLOR if not paused else MUTED_COLOR)
    if error:
        _label(screen, "AI unavailable", (x, area.y + 106), 21, BAD_COLOR)
        _label(screen, error[:38], (x, area.y + 132), 16, BAD_COLOR)
        return
    if trace is None:
        _label(screen, "Press A to enable AI", (x, area.y + 118), 21, MUTED_COLOR)
        _label(screen, "Press T to hide this panel", (x, area.y + 148), 17, MUTED_COLOR)
        return

    _label(screen, f"step {trace.step:03d}   confidence {trace.confidence:.1%}", (x, area.y + 104), 19)
    _label(screen, f"inference {trace.inference_ms:.2f} ms", (x, area.y + 128), 17, MUTED_COLOR)
    _label(screen, "CANDIDATE ACTIONS", (x, area.y + 170), 18, MUTED_COLOR)
    row_top = area.y + 196
    max_q = max((candidate.q_value for candidate in trace.candidates if candidate.legal), default=1.0)
    min_q = min((candidate.q_value for candidate in trace.candidates if candidate.legal), default=0.0)
    q_span = max_q - min_q or 1.0
    for index, candidate in enumerate(trace.candidates):
        y = row_top + index * 43
        selected = candidate.action == trace.selected_action
        row_rect = pygame.Rect(x - 6, y - 4, width + 12, 36)
        if selected:
            pygame.draw.rect(screen, (224, 236, 250), row_rect, border_radius=4)
        if selected:
            color = GOOD_COLOR if (candidate.heuristic_delta or 0) >= 0 else BAD_COLOR
        else:
            color = (150, 160, 172) if candidate.legal else (190, 195, 202)
        _label(screen, candidate.action.name, (x, y + 3), 18, color)
        _bar(screen, pygame.Rect(x + 76, y + 6, max(50, width - 168), 12), candidate.probability, color)
        q_fraction = (candidate.q_value - min_q) / q_span if candidate.legal else 0.0
        _bar(screen, pygame.Rect(x + 76, y + 22, max(50, width - 168), 5), q_fraction, GOOD_COLOR if candidate.legal else BAR_BACKGROUND)
        q_text = f"{candidate.q_value:+.2f}" if candidate.legal else "illegal"
        delta_text = f"d{candidate.heuristic_delta:+d}" if candidate.heuristic_delta is not None else ""
        _label(screen, delta_text, (x + width - 104, y + 7), 15, color if candidate.legal else MUTED_COLOR)
        _label(screen, q_text, (x + width - 56, y + 7), 16, TEXT_COLOR if candidate.legal else MUTED_COLOR)

    hidden_top = row_top + 4 * 43 + 22
    _label(screen, "HIDDEN ACTIVATION SUMMARY", (x, hidden_top), 17, MUTED_COLOR)
    hidden = trace.hidden_summary or (0.0,)
    maximum = max(abs(value) for value in hidden) or 1.0
    for index, value in enumerate(hidden):
        bar_height = int(min(42, abs(value) / maximum * 42))
        bar = pygame.Rect(x + index * 24, hidden_top + 24 + (42 - bar_height), 15, bar_height)
        pygame.draw.rect(screen, ACCENT_COLOR if value >= 0 else BAD_COLOR, bar, border_radius=2)
    _label(screen, "recent moves", (x, hidden_top + 82), 17, MUTED_COLOR)
    for index, action in enumerate(recent_actions[-5:]):
        _label(screen, action, (x, hidden_top + 105 + index * 20), 16, TEXT_COLOR)


def draw_action_overlay(
    screen: pygame.Surface,
    board_area: pygame.Rect,
    block_size: int,
    trace: DecisionTrace | None,
) -> None:
    if trace is None:
        return
    row, col = trace.blank_position
    start_x = board_area.x + (board_area.width - block_size * 4) // 2
    start_y = board_area.y + (board_area.height - block_size * 4) // 2
    start = pygame.Vector2(
        start_x + col * block_size + block_size / 2,
        start_y + row * block_size + block_size / 2,
    )
    delta_row, delta_col = ACTION_DELTAS[trace.selected_action]
    target = start + pygame.Vector2(delta_col * block_size, delta_row * block_size)
    selected = next(
        (candidate for candidate in trace.candidates if candidate.action == trace.selected_action),
        None,
    )
    overlay_color = (
        GOOD_COLOR
        if selected is not None and (selected.heuristic_delta or 0) >= 0
        else BAD_COLOR
        if selected is not None
        else ACCENT_COLOR
    )
    pygame.draw.line(screen, overlay_color, start, target, max(3, block_size // 14))
    direction = target - start
    if direction.length_squared() == 0:
        return
    direction.scale_to_length(block_size * 0.22)
    side = pygame.Vector2(-direction.y, direction.x) * 0.55
    tip = target
    pygame.draw.polygon(screen, overlay_color, [tip, tip - direction + side, tip - direction - side])
    target_rect = pygame.Rect(
        int(target.x - block_size / 2),
        int(target.y - block_size / 2),
        block_size,
        block_size,
    )
    pygame.draw.rect(screen, overlay_color, target_rect, max(2, block_size // 20))
