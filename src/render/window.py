from __future__ import annotations

import argparse
from pathlib import Path

import pygame
from pygame.locals import * # type: ignore
import logging
from config.basic import ROW_NUMBER, COL_NUMBER, LOGS_DIR
from config.render import (
    BACKGROUND_COLOR,
    BLOCK_COLOR,
    BLOCK_HIGHLIGHT_COLOR,
    TEXT_COLOR,
    HIGHLIGHT_TIME,
    WIN_HIGHLIGHT_COLOR,
    WIN_HIGHLIGHT_TIME,
    MOVE_TIME,
)
from board import Board
from board.board import generate_board
from ai.agent import DecisionAgent, DecisionTrace, load_agent
from ai.encoding import ACTION_DELTAS
from render.ai_panel import draw_action_overlay, draw_ai_panel

logger = logging.getLogger("game.render")

blockRects = []
highlight_blocks: dict[tuple[int, int], int] = {}  # 记录方格上次需要高亮的时刻。
move_blocks: tuple[tuple[int, int], tuple[int, int], int] = (
    (-1, -1),
    (-1, -1),
    -MOVE_TIME,
)  # 记录上一次移动的方块坐标和移动距离。


_MODEL_OPTIONS: tuple[tuple[str, Path], ...] = (
    ("第七代飞天数字华容道享受者", Path("models/第七代飞天数字华容道享受者.pt")),
    ("第六代无拐杖数字华容道之神", Path("models/第六代无拐杖数字华容道之神.pt")),
    ("第五代超级数字华容道之神", Path("models/第五代超级数字华容道之神.pt")),
    ("DQN 微调模型", Path("models/dqn.pt")),
    ("监督训练模型", Path("models/final.pt")),
)


def _available_model_options(model_path: Path) -> tuple[tuple[str, Path], ...]:
    """Return selectable model files, keeping a custom CLI path visible."""
    options = [option for option in _MODEL_OPTIONS if option[1].exists()]
    if model_path.exists() and all(path != model_path for _, path in options):
        options.insert(0, (model_path.stem, model_path))
    return tuple(options)


def _model_option_label(model_path: Path, options: tuple[tuple[str, Path], ...]) -> str:
    for label, path in options:
        if path == model_path:
            return label
    return model_path.stem


def draw_board(
    screen: pygame.Surface,
    board: Board,
    tick: int,
    *,
    win: bool = False,
    end_tick: int = 0,
    area: pygame.Rect | None = None,
) -> int:
    """绘制棋盘一帧；win=True 时叠加胜利对角线高亮波。返回 block_size。"""
    screen.fill(BACKGROUND_COLOR)  # Fill the screen with the background color
    # Draw game elements here
    draw_area = area or screen.get_rect()
    block_size = min(
        (
            draw_area.width // (COL_NUMBER + 2),
            draw_area.height // (ROW_NUMBER + 2),
        )
    )
    start_x = draw_area.x + (draw_area.width - (block_size * (COL_NUMBER))) // 2
    start_y = draw_area.y + (draw_area.height - (block_size * (ROW_NUMBER))) // 2
    blockRects.clear()
    for row in range(ROW_NUMBER):
        tmp = []
        for col in range(COL_NUMBER):
            if (col, row) == move_blocks[1] and tick - move_blocks[2] < MOVE_TIME:
                continue
            offset = (0, 0)
            text = str(board.board[row][col]) if board.board[row][col] != -1 else ""
            if (col, row) == move_blocks[0] and tick - move_blocks[2] < MOVE_TIME:
                time = (tick - move_blocks[2]) / MOVE_TIME
                direction = (
                    move_blocks[1][0] - move_blocks[0][0],
                    move_blocks[1][1] - move_blocks[0][1],
                )
                percent = 1 + (time - 1) ** 3
                offset = (
                    direction[0] * percent * block_size,
                    direction[1] * percent * block_size,
                )
                text = (
                    str(board.board[move_blocks[1][1]][move_blocks[1][0]])
                    if board.board[move_blocks[1][1]][move_blocks[1][0]] != -1
                    else ""
                )
                rect = pygame.Rect(
                    start_x + col * block_size,
                    start_y + row * block_size,
                    block_size,
                    block_size,
                )

                second = rect.move(
                    direction[0] * block_size, direction[1] * block_size
                )  # move 不修改原 rect，返回副本
                rect.union_ip(second)  # union_ip 修改原 rect
                pygame.draw.rect(
                    screen,
                    BLOCK_COLOR,
                    rect,
                    2,
                )
            rect = pygame.Rect(
                start_x + col * block_size,
                start_y + row * block_size,
                block_size,
                block_size,
            ).move(*offset)
            tmp.append(rect)
            font = pygame.font.Font(None, int(block_size * 0.5))
            text_surface = font.render(text, True, BLOCK_COLOR)
            text_rect = text_surface.get_rect(center=rect.center)
            if (
                tick - highlight_blocks.get((row, col), -HIGHLIGHT_TIME)
                < HIGHLIGHT_TIME
            ):
                old_color = pygame.Color(BLOCK_HIGHLIGHT_COLOR)
                new_color = pygame.Color(BACKGROUND_COLOR)
                color = pygame.Color(
                    int(
                        (new_color.r - old_color.r)
                        * (tick - highlight_blocks[(row, col)])
                        / HIGHLIGHT_TIME
                        + old_color.r
                    ),
                    int(
                        (new_color.g - old_color.g)
                        * (tick - highlight_blocks[(row, col)])
                        / HIGHLIGHT_TIME
                        + old_color.g
                    ),
                    int(
                        (new_color.b - old_color.b)
                        * (tick - highlight_blocks[(row, col)])
                        / HIGHLIGHT_TIME
                        + old_color.b
                    ),
                    int(
                        (new_color.a - old_color.a)
                        * (tick - highlight_blocks[(row, col)])
                        / HIGHLIGHT_TIME
                        + old_color.a
                    ),
                )
                pygame.draw.rect(screen, color, rect)
            if win:
                if (
                    (col + row) * WIN_HIGHLIGHT_TIME
                    < (tick - end_tick)
                    % (WIN_HIGHLIGHT_TIME * (ROW_NUMBER + COL_NUMBER + 1))
                    < (col + row + 1) * WIN_HIGHLIGHT_TIME
                ):
                    old_color = pygame.Color(BACKGROUND_COLOR)
                    new_color = pygame.Color(WIN_HIGHLIGHT_COLOR)
                    percent = (
                        1
                        - (
                            2
                            * (
                                (
                                    (tick - end_tick)
                                    % (
                                        WIN_HIGHLIGHT_TIME
                                        * (ROW_NUMBER + COL_NUMBER + 1)
                                    )
                                    - (col + row) * WIN_HIGHLIGHT_TIME
                                )
                                / WIN_HIGHLIGHT_TIME
                            )
                            - 1
                        )
                        ** 2
                    )
                    color = pygame.Color(
                        int((new_color.r - old_color.r) * percent + old_color.r),
                        int((new_color.g - old_color.g) * percent + old_color.g),
                        int((new_color.b - old_color.b) * percent + old_color.b),
                        int((new_color.a - old_color.a) * percent + old_color.a),
                    )
                    pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLOCK_COLOR, rect, 2)  # Draw the block border
            screen.blit(text_surface, text_rect)

        blockRects.append(tmp)
    return block_size


def pick_block(
    mouse_pos: tuple[int, int], old_block_num: tuple[int, int]
) -> tuple[int, int]:
    """返回鼠标悬停的方块坐标，未命中则返回 (-1, -1)。"""
    for i, rects in enumerate(blockRects):
        for j, rect in enumerate(rects):
            if rect.collidepoint(mouse_pos):
                if old_block_num != (j, i):
                    logger.info(f"Move to block {j},{i} at position {mouse_pos}")
                return (j, i)
    return (-1, -1)


def _board_state(board: Board) -> tuple[int, ...]:
    return tuple(value for line in board.board for value in line)


def _reset_random_board(board: Board) -> None:
    board.board = generate_board(COL_NUMBER, ROW_NUMBER)


def _reset_goal_board(board: Board) -> None:
    board.board = [
        [i for i in range(j * COL_NUMBER + 1, (j + 1) * COL_NUMBER + 1)]
        for j in range(ROW_NUMBER)
    ]
    board.board[ROW_NUMBER - 1][COL_NUMBER - 1] = -1


def _reset_agent_history(agent: DecisionAgent | None) -> None:
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()


def start(board: Board, *, model_path: Path = Path("models/第七代飞天数字华容道享受者.pt"), smoke_test: bool = False):
    global move_blocks, blockRects, highlight_blocks
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode((1180, 720), pygame.RESIZABLE)
    pygame.display.set_caption("DigitalHuarongPass")
    if smoke_test:
        draw_board(screen, board, pygame.time.get_ticks(), area=screen.get_rect())
        pygame.display.flip()
        pygame.quit()
        return

    move_blocks = ((-1, -1), (-1, -1), -MOVE_TIME)
    block_num = (-1, -1)
    ai_enabled = False
    ai_paused = False
    ai_step_requested = False
    panel_visible = True
    agent: DecisionAgent | None = None
    model_options = _available_model_options(model_path)
    model_name = _model_option_label(model_path, model_options)
    model_dropdown_open = False
    selector_rect = pygame.Rect(0, 0, 0, 0)
    option_rects: tuple[pygame.Rect, ...] = ()
    ai_error: str | None = None
    trace: DecisionTrace | None = None
    recent_actions: list[str] = []
    win = False
    end_tick = 0
    running = True
    while running:
        tick = pygame.time.get_ticks()
        panel_width = min(380, max(0, screen.get_width() - 520)) if panel_visible else 0
        show_panel = panel_width >= 220
        if not show_panel:
            panel_width = 0
        board_area = pygame.Rect(0, 0, max(1, screen.get_width() - panel_width), screen.get_height())
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                if show_panel and selector_rect.collidepoint(event.pos):
                    model_dropdown_open = not model_dropdown_open
                    continue
                if model_dropdown_open:
                    selected_index = next(
                        (index for index, rect in enumerate(option_rects) if rect.collidepoint(event.pos)),
                        None,
                    )
                    if selected_index is not None:
                        selected_label, selected_path = model_options[selected_index]
                        was_enabled = ai_enabled
                        try:
                            loaded_agent, metadata = load_agent(selected_path, use_guard=False)
                            agent = loaded_agent  # type: ignore[assignment]
                            _reset_agent_history(agent)
                            model_path = selected_path
                            model_name = str(metadata.get("display_name") or selected_label)
                            ai_error = None
                            trace = None
                            recent_actions.clear()
                            ai_enabled = was_enabled
                            ai_paused = False
                        except (OSError, ValueError, RuntimeError) as exc:
                            ai_error = str(exc)
                            ai_enabled = False
                            agent = None
                    model_dropdown_open = False
                    continue
                if ai_enabled or win:
                    continue
                if tick - move_blocks[2] < MOVE_TIME:
                    continue
                if block_num != (-1, -1):
                    result = board.dealWithSwap(block_num)
                    if result is not None:
                        move_blocks = (block_num, (result[1], result[0]), tick)
                        trace = None
                    if board.checkWin():
                        logger.info("You win!")
                        win = True
                        end_tick = tick
            elif event.type == MOUSEMOTION:
                block_num = pick_block(event.pos, block_num)
            elif event.type == KEYDOWN:
                logger.info(f"Key pressed: {pygame.key.name(event.key)}")
                if event.key == K_a and not win:
                    if ai_enabled:
                        ai_enabled = False
                        ai_paused = False
                        trace = None
                    else:
                        try:
                            # The seventh-generation model is fully neural at runtime.
                            # A* remains an offline training teacher only.
                            loaded_agent, metadata = load_agent(model_path, use_guard=False)
                            agent = loaded_agent  # type: ignore[assignment]
                            _reset_agent_history(agent)
                            model_name = str(
                                metadata.get("display_name", metadata.get("stage", model_path.name))
                            )
                            ai_error = None
                            ai_enabled = True
                            ai_paused = False
                            trace = None
                        except (OSError, ValueError, RuntimeError) as exc:
                            ai_error = str(exc)
                            ai_enabled = False
                            agent = None
                elif event.key == K_SPACE and ai_enabled and not win:
                    if ai_paused:
                        ai_step_requested = True
                    else:
                        ai_paused = True
                elif event.key == K_t:
                    panel_visible = not panel_visible
                elif event.key == K_r and not win:
                    _reset_random_board(board)
                    _reset_agent_history(agent)
                    move_blocks = ((-1, -1), (-1, -1), tick - MOVE_TIME)
                    trace = None
                    recent_actions.clear()
                elif event.key == K_ESCAPE:
                    _reset_goal_board(board)
                    _reset_agent_history(agent)
                    move_blocks = ((-1, -1), (-1, -1), tick - MOVE_TIME)
                    trace = None
                    win = False
                elif win:
                    running = False

        can_move = tick - move_blocks[2] >= MOVE_TIME
        should_step = ai_enabled and agent is not None and not win and can_move and (
            not ai_paused or ai_step_requested
        )
        if should_step:
            current_state = _board_state(board)
            trace = agent.decide(current_state, step=trace.step + 1 if trace else 0)
            blank_row, blank_col = trace.blank_position
            delta_row, delta_col = ACTION_DELTAS[trace.selected_action]
            target = (blank_col + delta_col, blank_row + delta_row)
            old_blank = board.dealWithSwap(target)
            if old_blank is not None:
                move_blocks = (target, (old_blank[1], old_blank[0]), tick)
                recent_actions.append(trace.selected_action.name)
                recent_actions[:] = recent_actions[-8:]
            ai_step_requested = False
            if board.checkWin():
                win = True
                end_tick = tick

        if tick - move_blocks[2] >= MOVE_TIME and block_num != (-1, -1):
            highlight_blocks[(block_num[1], block_num[0])] = tick
        block_size = draw_board(screen, board, tick, win=win, end_tick=end_tick, area=board_area)
        if ai_enabled:
            draw_action_overlay(screen, board_area, block_size, trace)
        if show_panel:
            selector_rect, option_rects = draw_ai_panel(
                screen,
                pygame.Rect(screen.get_width() - panel_width, 0, panel_width, screen.get_height()),
                trace,
                model_name=model_name,
                paused=ai_paused,
                error=ai_error,
                recent_actions=recent_actions,
                model_options=model_options,
                model_dropdown_open=model_dropdown_open,
            )
        else:
            selector_rect = pygame.Rect(0, 0, 0, 0)
            option_rects = ()
        if win:
            font = pygame.font.Font(None, int(block_size * 1.5))
            text_surface = font.render("You win!", True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(board_area.centerx, board_area.centery - block_size))
            screen.blit(text_surface, text_rect)
        pygame.display.flip()
    pygame.quit()


def main(*, model_path: Path = Path("models/第七代飞天数字华容道享受者.pt"), smoke_test: bool = False) -> None:
    """控制台入口：建立棋盘、初始化日志并启动游戏窗口。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("game")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
    logger_.setLevel(logging.DEBUG)
    logger_.addHandler(logging.StreamHandler())
    logger_.addHandler(logging.FileHandler(LOGS_DIR / "game.log", mode="w"))
    for handler in logger_.handlers:
        handler.setFormatter(formatter)

    start(Board(COL_NUMBER, ROW_NUMBER), model_path=model_path, smoke_test=smoke_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DigitalHuarongPass")
    parser.add_argument("--model", type=Path, default=Path("models/第七代飞天数字华容道享受者.pt"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(model_path=args.model, smoke_test=args.smoke_test)
