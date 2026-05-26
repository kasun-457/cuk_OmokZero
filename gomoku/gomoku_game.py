"""11x11 Gomoku game logic (5 in a row wins)."""
from copy import deepcopy

import numpy as np

from game import Game

BOARD_SIZE = 11
WIN_LEN = 5

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class GomokuGame(Game):
    """11x11 Gomoku.

    Board values: 0 = empty, 1 = black (first player), -1 = white (second player).
    state shape: (4, BOARD_SIZE, BOARD_SIZE)
      ch0: current player's stones
      ch1: opponent's stones
      ch2: empty squares
      ch3: last-move indicator
    """

    def __init__(self):
        self.row = BOARD_SIZE
        self.column = BOARD_SIZE
        self.action_size = BOARD_SIZE * BOARD_SIZE
        self.current_player = 1
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.last_move = None
        self._winner = 0          # cached winner after game ends
        self._game_over = False

    # ------------------------------------------------------------------
    # Game interface
    # ------------------------------------------------------------------

    def clone(self):
        g = GomokuGame()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        g._winner = self._winner
        g._game_over = self._game_over
        return g

    def play_action(self, action):
        """action: integer index = row * BOARD_SIZE + col."""
        r, c = divmod(action, BOARD_SIZE)
        assert self.board[r, c] == 0, f"Square ({r},{c}) already occupied"
        self.board[r, c] = self.current_player
        self.last_move = action
        if self._check_win(r, c, self.current_player):
            self._winner = self.current_player
            self._game_over = True
        elif not np.any(self.board == 0):
            self._winner = 0
            self._game_over = True
        self.current_player = -self.current_player

    def get_valid_moves(self):
        """Returns boolean mask of length action_size."""
        return (self.board.flatten() == 0).astype(np.float32)

    def check_game_over(self):
        """Returns (game_over: bool, value: int).

        value is from the perspective of the player whose turn it was BEFORE
        the last move (i.e., self.current_player after play_action has flipped).
        win=1, loss=-1, draw=0.
        """
        if self._game_over:
            return True, self._winner * (-self.current_player)
        return False, 0

    @property
    def state(self):
        """4-channel float32 representation of the board."""
        cp = self.current_player
        s = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        s[0] = (self.board == cp).astype(np.float32)
        s[1] = (self.board == -cp).astype(np.float32)
        s[2] = (self.board == 0).astype(np.float32)
        if self.last_move is not None:
            r, c = divmod(self.last_move, BOARD_SIZE)
            s[3, r, c] = 1.0
        return s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_win(self, r, c, player):
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr, nc] == player:
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= WIN_LEN:
                return True
        return False

    def print_board(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        header = "   " + " ".join(f"{c:2d}" for c in range(BOARD_SIZE))
        print(header)
        for r in range(BOARD_SIZE):
            row_str = f"{r:2d} " + "  ".join(symbols[self.board[r, c]] for c in range(BOARD_SIZE))
            print(row_str)
        print()
