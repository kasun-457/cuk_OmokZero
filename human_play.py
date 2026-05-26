"""Human vs AI terminal interface for 11x11 Gomoku."""
from mcts import MonteCarloTreeSearch, TreeNode
from config import CFG
from gomoku.gomoku_game import BOARD_SIZE


class HumanPlay:
    def __init__(self, game, net):
        self.game = game
        self.net = net

    def play(self):
        print("=" * 50)
        print("  11x11 오목 (Gomoku)  —  Human vs AlphaZero AI")
        print("=" * 50)
        print("입력 형식: 행 열  (예: 5 5)  |  'quit' 으로 종료\n")

        mcts = MonteCarloTreeSearch(self.net)
        game = self.game.clone()
        node = TreeNode()

        go_first = input("먼저 두시겠습니까? (y/n): ").strip().lower()
        human_player = 1 if go_first == "y" else -1
        human_sym = "X(흑)" if human_player == 1 else "O(백)"
        ai_sym = "O(백)" if human_player == 1 else "X(흑)"
        print(f"\n당신: {human_sym}  |  AI: {ai_sym}\n")

        game.print_board()

        while True:
            game_over, value = game.check_game_over()
            if game_over:
                self._print_result(value, human_player, game)
                return

            if game.current_player == human_player:
                action = self._get_human_move(game)
                if action is None:
                    print("게임을 종료합니다.")
                    return
                best_child = TreeNode()
                best_child.action = action
            else:
                print("AI가 생각 중입니다...")
                best_child = mcts.search(game, node, CFG.temp_final)
                r, c = divmod(best_child.action, BOARD_SIZE)
                print(f"AI 착수: ({r}, {c})\n")

            game.play_action(best_child.action)
            best_child.parent = None
            node = best_child
            game.print_board()

    # ------------------------------------------------------------------

    def _get_human_move(self, game):
        valid = game.get_valid_moves()
        while True:
            raw = input("당신의 착수 (행 열): ").strip()
            if raw.lower() == "quit":
                return None
            try:
                parts = raw.split()
                r, c = int(parts[0]), int(parts[1])
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    print(f"  범위를 벗어났습니다. 0~{BOARD_SIZE-1} 사이로 입력하세요.")
                    continue
                action = r * BOARD_SIZE + c
                if valid[action] == 0:
                    print("  이미 돌이 있는 위치입니다.")
                    continue
                return action
            except (ValueError, IndexError):
                print("  형식이 잘못됐습니다. 예: 5 5")

    def _print_result(self, value, human_player, game):
        print("=" * 50)
        if value == 0:
            print("무승부!")
        elif value * (-game.current_player) == human_player:
            print("축하합니다! 당신이 이겼습니다!")
        else:
            print("AI가 이겼습니다. 다음에 다시 도전하세요!")
        print("=" * 50)
