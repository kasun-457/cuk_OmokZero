"""Abstract base class for 2-player board games."""


class Game:
    def clone(self):
        raise NotImplementedError

    def play_action(self, action):
        raise NotImplementedError

    def get_valid_moves(self):
        raise NotImplementedError

    def check_game_over(self):
        raise NotImplementedError

    def print_board(self):
        raise NotImplementedError
