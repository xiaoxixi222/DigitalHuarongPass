import random
import logging
import pygame

logger = logging.getLogger("game.board")

#deepseek加的这个注释句号有点多啊
class Board:
    """棋盘类，表示数字华容道的棋盘。

    Attributes:
        col (int): 棋盘的列数。
        row (int): 棋盘的行数。
        board (list[list[int]]): 二维列表，表示棋盘上的数字和空格。
    """

    def __init__(self, col: int, row: int):
        """初始化棋盘。

        Args:
            col (int): 棋盘的列数。
            row (int): 棋盘的行数。
        """
        self.col = col
        self.row = row
        self.board = generate_board(col, row)
        logger.debug(f"Initialized board: {self.board}")
        logger.info(f"Board created with dimensions {col}x{row}")

    def dealWithSwap(self, pos: pygame.Vector2) -> pygame.Vector2 | None:
        """处理两个位置的交换。
        自动寻找挨着他的空白格，并完成与空白格的交换此时输出空白格的坐标(用于制作动画),若点击的为空白格或点击格周围没有空白格，则输出false

        Args:
            pos (pygame.Vector2): 被点击格子的坐标，x 为行索引，y 为列索引（与点击处理得到的 block_num=(行, 列) 一致）。

        Returns:
            pygame.Vector2 | None: 如果交换成功，返回空白格交换后的坐标（x 为行索引，y 为列索引）；否则返回 None。
        """
        row, col = int(pos.x), int(pos.y)
        if not (0 <= row < self.row and 0 <= col < self.col):
            return None
        if self.board[row][col] == -1:
            return None

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            blank_row, blank_col = row + dr, col + dc
            if 0 <= blank_row < self.row and 0 <= blank_col < self.col:
                if self.board[blank_row][blank_col] == -1:
                    self.board[row][col], self.board[blank_row][blank_col] = (
                        self.board[blank_row][blank_col],
                        self.board[row][col],
                    )
                    logger.debug(f"Swapped ({row},{col}) with blank ({blank_row},{blank_col})")
                    return pygame.Vector2(blank_row, blank_col)

        return None

    def checkWin(self) -> bool:
        """检查当前棋盘是否处于胜利状态。

        Returns:
            bool: 如果棋盘处于胜利状态，返回 True；否则返回 False。
        """
        flat = [v for line in self.board for v in line]
        expected = list(range(1, self.col * self.row)) + [-1]
        return flat == expected


def generate_board(col: int, row: int) -> list[list[int]]:
    """生成一个保证可解的随机数字华容道棋盘。

    返回 row 行 x col 列的二维列表，数字为随机打乱的 1..col*row-1，-1 表示空格。
    """
    if col < 2 or row < 2:
        raise ValueError("col 和 row 至少为 2")

    tiles = list(range(1, col * row)) + [-1]
    random.shuffle(tiles)
    board = [tiles[r * col : (r + 1) * col] for r in range(row)]

    flat = [v for line in board for v in line]
    nums = [v for v in flat if v != -1]
    inversions = sum(
        1
        for i in range(len(nums))
        for j in range(i + 1, len(nums))
        if nums[i] > nums[j]
    )
    blank_row_from_bottom = row - flat.index(-1) // col
    if col % 2 == 1:
        solvable = inversions % 2 == 0
    else:
        solvable = (inversions + blank_row_from_bottom) % 2 == 1
#不可解情况(这破玩意怎么这么难搞)
    if not solvable:
        positions = [
            (r, c) for r in range(row) for c in range(col) if board[r][c] != -1
        ]
        (r1, c1), (r2, c2) = random.sample(positions, 2)
        board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]

    return board
