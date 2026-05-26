"""Evaluation: new model vs best model via self-play."""
from config import CFG
from mcts import MonteCarloTreeSearch, TreeNode


class Evaluate:
    def __init__(self, current_mcts, eval_mcts, game):
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts
        self.game = game

    def evaluate(self):
        wins = losses = 0

        for i in range(CFG.num_eval_games):
            game = self.game.clone()
            node = TreeNode()
            game_over = False

            # Alternate which side the new model plays
            new_model_player = 1 if i % 2 == 0 else -1

            while not game_over:
                if game.current_player == new_model_player:
                    best_child = self.current_mcts.search(game, node, CFG.temp_final)
                else:
                    best_child = self.eval_mcts.search(game, node, CFG.temp_final)

                game.play_action(best_child.action)
                best_child.parent = None
                node = best_child
                game_over, value = game.check_game_over()

            # value is from last-mover's perspective; adjust to new_model_player
            # After game ends current_player has already swapped, so:
            if value == 0:
                pass  # draw — neither wins nor losses
            elif value * new_model_player * (-game.current_player) > 0:
                wins += 1
            else:
                losses += 1

            print(f"  Eval game {i+1}: {'win' if value == 0 else ('win' if value * new_model_player * (-game.current_player) > 0 else 'loss')}")

        return wins, losses
