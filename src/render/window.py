import pygame
from pygame.locals import *
import logging
from config.basic import ROW_NUMBER, COL_NUMBER, LOGS_DIR
from config.render import BACKGROUND_COLOR, BLOCK_COLOR, TEXT_COLOR
from board import Board

logger = logging.getLogger("game.render")

blockRects = []


def start(board: Board):
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("DigitalHuarongPass")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                # logger.info(f"Mouse button pressed at position {event.pos}")
                if event.button != 1:
                    continue
                mouse_pos = event.pos
                block_num = (-1, -1)
                for i, rects in enumerate(blockRects):
                    for j, rect in enumerate(rects):
                        if rect.collidepoint(mouse_pos):
                            logger.info(
                                f"Block {j},{i} clicked at position {mouse_pos}"
                            )
                            block_num = (j, i)

                            break
                    if block_num != (-1, -1):
                        break
                if block_num != (-1, -1):
                    logger.info(f"Swapping block {block_num}")
                    logger.info(
                        f"Result: {board.dealWithSwap(pygame.Vector2(block_num[0], block_num[1]))}"
                    )
                    if board.checkWin():
                        logger.info("You win!")
                        running = False
            elif event.type == KEYDOWN:
                logger.info(f"Key pressed: {pygame.key.name(event.key)}")
                if event.key == K_ESCAPE:
                    board.board = [[i for i in range(j * COL_NUMBER+1, (j + 1) * COL_NUMBER+1)] for j in range(ROW_NUMBER)]
                    board.board[ROW_NUMBER - 1][COL_NUMBER - 1] = -1
                    logger.info("Resetting the board to the initial state.")

        screen.fill(BACKGROUND_COLOR)  # Fill the screen with the background color
        # Draw game elements here
        block_size = min(
            (
                screen.get_width() // (COL_NUMBER + 2),
                screen.get_height() // (ROW_NUMBER + 2),
            )
        )
        start_x = (screen.get_width() - (block_size * (COL_NUMBER))) // 2
        start_y = (screen.get_height() - (block_size * (ROW_NUMBER))) // 2
        blockRects.clear()
        for row in range(ROW_NUMBER):
            tmp = []
            for col in range(COL_NUMBER):
                rect = pygame.Rect(
                    start_x + col * block_size,
                    start_y + row * block_size,
                    block_size,
                    block_size,
                )
                tmp.append(rect)
                text = str(board.board[row][col]) if board.board[row][col] != -1 else ""
                font = pygame.font.Font(None, int(block_size * 0.5))
                text_surface = font.render(text, True, BLOCK_COLOR)
                text_rect = text_surface.get_rect(center=rect.center)
                pygame.draw.rect(screen, BLOCK_COLOR, rect, 2)  # Draw the block border
                screen.blit(text_surface, text_rect)

            blockRects.append(tmp)

        pygame.display.flip()  # Update the display
    old_screen = screen.copy()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                pygame.quit()
                return
        screen.fill(BACKGROUND_COLOR)
        screen.blit(old_screen, (0, 0))
        screen.blit(
            pygame.font.Font(None, 72).render("You Win!", True, TEXT_COLOR),
            (screen.get_width() // 2 - 100, screen.get_height() // 2 - 36)
        )
        screen.blit(
            pygame.font.Font(None, 36).render("Press ESC to exit.", True, TEXT_COLOR),
            (screen.get_width() // 2 - 100, screen.get_height() // 2 + 36)
        )
        pygame.display.flip()
    pygame.quit()


def main() -> None:
    """控制台入口：建立棋盘、初始化日志并启动游戏窗口。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("game")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
    logger_.setLevel(logging.DEBUG)
    logger_.addHandler(logging.StreamHandler())
    logger_.addHandler(logging.FileHandler(LOGS_DIR / "game.log", mode="w"))
    for handler in logger_.handlers:
        handler.setFormatter(formatter)

    start(Board(COL_NUMBER, ROW_NUMBER))
