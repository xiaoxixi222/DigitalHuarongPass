from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

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
ACTION_LABELS = {
    "UP": "上",
    "DOWN": "下",
    "LEFT": "左",
    "RIGHT": "右",
}
MODEL_LABELS = {
    "sixth-generation": "第六代无拐杖数字华容道之神",
    "第七代飞天数字华容道享受者.pt": "第七代飞天数字华容道享受者",
    "sixth-generation-fold": "第六代深度折模型",
    "cross-validation-ensemble": "第五代超级数字华容道之神",
    "cross-validation-fold": "交叉验证模型",
    "imitation-validation": "监督验证模型",
    "imitation-final": "监督训练模型",
    "dqn": "强化学习微调模型",
    "neural solver": "神经网络求解器",
    "untrained": "未训练模型",
    "第五代超级数字华容道之神.pt": "第五代超级数字华容道之神",
    "not loaded": "未加载模型",
}


@lru_cache(maxsize=None)
def _font(size: int) -> pygame.font.Font:
    for font_path in (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(font_path).exists():
            return pygame.font.Font(font_path, size)
    return pygame.font.SysFont("PingFang SC", size) or pygame.font.Font(None, size)


def _action_label(action: str) -> str:
    return ACTION_LABELS.get(action, action)


def _model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def _label(surface: pygame.Surface, text: str, position: tuple[int, int], size: int = 20, color=TEXT_COLOR) -> None:
    surface.blit(_font(size).render(text, True, color), position)


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
    model_name: str = "神经网络求解器",
    paused: bool = False,
    error: str | None = None,
    recent_actions: Sequence[str] = (),
    model_options: Sequence[tuple[str, Path]] = (),
    model_dropdown_open: bool = False,
) -> tuple[pygame.Rect, tuple[pygame.Rect, ...]]:
    pygame.draw.rect(screen, PANEL_BACKGROUND, area)
    pygame.draw.line(screen, PANEL_BORDER, area.topleft, area.bottomleft, 2)
    x = area.x + 18
    width = area.width - 36
    _label(screen, "人工智能决策轨迹", (x, area.y + 16), 22)
    selector_rect = pygame.Rect(x, area.y + 43, width, 32)
    pygame.draw.rect(screen, (255, 255, 255), selector_rect, border_radius=4)
    pygame.draw.rect(screen, PANEL_BORDER, selector_rect, 1, border_radius=4)
    selected_label = _model_label(model_name)
    if len(selected_label) > 27:
        selected_label = selected_label[:26] + "…"
    _label(screen, f"模型：{selected_label}", (selector_rect.x + 10, selector_rect.y + 7), 16, TEXT_COLOR)
    arrow_x = selector_rect.right - 19
    arrow_y = selector_rect.centery
    arrow = (
        [(arrow_x - 5, arrow_y - 2), (arrow_x + 5, arrow_y - 2), (arrow_x, arrow_y + 4)]
        if not model_dropdown_open
        else [(arrow_x - 5, arrow_y + 3), (arrow_x + 5, arrow_y + 3), (arrow_x, arrow_y - 4)]
    )
    pygame.draw.polygon(screen, MUTED_COLOR, arrow)
    option_rects: list[pygame.Rect] = []
    if model_dropdown_open:
        for index, (_label_text, _path) in enumerate(model_options):
            option_rect = pygame.Rect(
                selector_rect.x,
                selector_rect.bottom + index * 30,
                selector_rect.width,
                30,
            )
            option_rects.append(option_rect)

    def draw_options() -> None:
        for option_rect, (label, _path) in zip(option_rects, model_options):
            pygame.draw.rect(screen, (255, 255, 255), option_rect)
            pygame.draw.rect(screen, PANEL_BORDER, option_rect, 1)
            option_label = label if len(label) <= 29 else label[:28] + "…"
            _label(screen, option_label, (option_rect.x + 10, option_rect.y + 6), 15, TEXT_COLOR)
    status = "已暂停 / 单步" if paused else "运行中"
    _label(screen, f"状态：{status}", (x, area.y + 84), 17, ACCENT_COLOR if not paused else MUTED_COLOR)
    if error:
        _label(screen, "人工智能不可用", (x, area.y + 126), 19, BAD_COLOR)
        _label(screen, f"错误：{error[:34]}", (x, area.y + 152), 16, BAD_COLOR)
        draw_options()
        return selector_rect, tuple(option_rects)
    if trace is None:
        _label(screen, "按 A 键启动人工智能", (x, area.y + 138), 19, MUTED_COLOR)
        _label(screen, "按 T 键隐藏面板", (x, area.y + 168), 17, MUTED_COLOR)
        draw_options()
        return selector_rect, tuple(option_rects)

    source = (
        "A* 循环保护"
        if trace.decision_source == "astar-loop-break"
        else "A* 保护动作"
        if trace.fallback_used
        else "神经网络（防振荡）"
        if trace.decision_source == "neural-anti-loop"
        else "神经网络"
    )
    compact = width < 280
    if compact:
        _label(screen, f"步数 {trace.step:03d}", (x, area.y + 124), 18)
        _label(screen, f"置信度 {trace.confidence:.1%}", (x, area.y + 146), 17)
        source_y = area.y + 168
        agreement_y = area.y + 189
        inference_y = area.y + 210
        candidate_y = area.y + 242
        row_top = area.y + 268
    else:
        _label(screen, f"步数 {trace.step:03d}   置信度 {trace.confidence:.1%}", (x, area.y + 124), 19)
        source_y = area.y + 148
        agreement_y = area.y + 169
        inference_y = area.y + 190
        candidate_y = area.y + 222
        row_top = area.y + 248
    _label(screen, source, (x, source_y), 16, BAD_COLOR if trace.fallback_used else MUTED_COLOR)
    _label(screen, f"折模型一致性 {trace.fold_agreement:.0%}", (x, agreement_y), 16, BAD_COLOR if trace.fallback_used else MUTED_COLOR)
    _label(screen, f"推理耗时 {trace.inference_ms:.2f} 毫秒", (x, inference_y), 16, MUTED_COLOR)
    _label(screen, "候选动作", (x, candidate_y), 18, MUTED_COLOR)
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
        _label(screen, _action_label(candidate.action.name), (x, y + 3), 18, color)
        delta_text = f"启发式{candidate.heuristic_delta:+d}" if candidate.heuristic_delta is not None else ""
        q_text = f"{candidate.q_value:+.2f}" if candidate.legal else "非法"
        value_font_size = 15 if width >= 280 else 13
        q_font_size = 16 if width >= 280 else 14
        right_edge = area.right - 18
        q_width = _font(q_font_size).size(q_text)[0]
        delta_width = _font(value_font_size).size(delta_text)[0]
        q_x = right_edge - q_width
        delta_x = q_x - 8 - delta_width
        bar_x = x + (76 if width >= 280 else 40)
        bar_width = max(1, delta_x - 10 - bar_x)
        _bar(screen, pygame.Rect(bar_x, y + 6, bar_width, 12), candidate.probability, color)
        q_fraction = (candidate.q_value - min_q) / q_span if candidate.legal else 0.0
        _bar(screen, pygame.Rect(bar_x, y + 22, bar_width, 5), q_fraction, GOOD_COLOR if candidate.legal else BAR_BACKGROUND)
        _label(screen, delta_text, (delta_x, y + 7), value_font_size, color if candidate.legal else MUTED_COLOR)
        _label(screen, q_text, (q_x, y + 7), q_font_size, TEXT_COLOR if candidate.legal else MUTED_COLOR)

    hidden_top = row_top + 4 * 43 + 22
    _label(screen, "隐藏层激活摘要", (x, hidden_top), 17, MUTED_COLOR)
    hidden = trace.hidden_summary or (0.0,)
    maximum = max(abs(value) for value in hidden) or 1.0
    for index, value in enumerate(hidden):
        bar_height = int(min(42, abs(value) / maximum * 42))
        bar = pygame.Rect(x + index * 24, hidden_top + 24 + (42 - bar_height), 15, bar_height)
        pygame.draw.rect(screen, ACCENT_COLOR if value >= 0 else BAD_COLOR, bar, border_radius=2)
    _label(screen, "最近动作", (x, hidden_top + 82), 17, MUTED_COLOR)
    for index, action in enumerate(recent_actions[-5:]):
        _label(screen, _action_label(action), (x, hidden_top + 105 + index * 20), 16, TEXT_COLOR)
    draw_options()
    return selector_rect, tuple(option_rects)


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
