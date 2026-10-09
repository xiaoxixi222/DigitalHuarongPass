import pygame
from pathlib import Path

from board import Board
from render.window import _available_model_options, _model_option_label, draw_board


def test_draw_board_accepts_a_bounded_play_area():
    pygame.init()
    screen = pygame.Surface((1180, 720))
    board = Board(4, 4)

    block_size = draw_board(screen, board, 0, area=pygame.Rect(0, 0, 800, 720))

    assert block_size > 0
    pygame.quit()


def test_model_dropdown_lists_available_named_models():
    options = _available_model_options(Path("models/第七代飞天数字华容道享受者.pt"))
    labels = [label for label, _path in options]

    assert "第七代飞天数字华容道享受者" in labels
    assert _model_option_label(Path("models/第七代飞天数字华容道享受者.pt"), options) == "第七代飞天数字华容道享受者"
