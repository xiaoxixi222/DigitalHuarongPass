import pygame

from board import Board
from render.window import draw_board


def test_draw_board_accepts_a_bounded_play_area():
    pygame.init()
    screen = pygame.Surface((1180, 720))
    board = Board(4, 4)

    block_size = draw_board(screen, board, 0, area=pygame.Rect(0, 0, 800, 720))

    assert block_size > 0
    pygame.quit()
