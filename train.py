"""Self-play training loop for 11x11 Gomoku AlphaZero."""
import numpy as np
from copy import deepcopy

from config import CFG
from mcts import MonteCarloTreeSearch, TreeNode
from neural_net import NeuralNetworkWrapper
from evaluate import Evaluate


class Train:
    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.eval_net = NeuralNetworkWrapper(game)

    def start(self):
        for iteration in range(1, CFG.num_iterations + 1):
            print(f"\n=== Iteration {iteration}/{CFG.num_iterations} ===")

            training_data = []
            for g in range(1, CFG.num_games + 1):
                print(f"  Self-play game {g}/{CFG.num_games}", end="\r")
                game = self.game.clone()
                self._play_game(game, training_data)
            print()

            self.net.save_model()
            self.eval_net.load_model()
            self.net.train(training_data)

            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)
            evaluator = Evaluate(current_mcts, eval_mcts, self.game)
            wins, losses = evaluator.evaluate()

            played = wins + losses
            win_rate = wins / played if played > 0 else 0.0
            print(f"Eval  wins={wins}  losses={losses}  win_rate={win_rate:.2f}")

            if win_rate > CFG.eval_win_rate:
                print("New best model saved.")
                self.net.save_model("best_model")
            else:
                print("Previous model kept.")
                self.net.load_model()

    # ------------------------------------------------------------------

    def _play_game(self, game, training_data):
        mcts = MonteCarloTreeSearch(self.net)
        node = TreeNode()
        self_play_data = []
        move_count = 0

        while True:
            temp = CFG.temp_init if move_count < CFG.temp_thresh else CFG.temp_final
            best_child = mcts.search(game, node, temp)

            # Build policy vector from visit counts
            pi = np.zeros(game.action_size, dtype=np.float32)
            total_visits = sum(c.Nsa for c in node.children)
            if total_visits > 0:
                for child in node.children:
                    pi[child.action] = child.Nsa / total_visits

            self_play_data.append((deepcopy(game.state), pi, 0.0))

            game.play_action(best_child.action)
            move_count += 1

            game_over, value = game.check_game_over()
            if game_over:
                # Assign values: alternate signs going backward
                for i, (s, p, _) in enumerate(reversed(self_play_data)):
                    v = value * ((-1) ** i)
                    self._augment(s, p, v, training_data, game.row, game.column)
                return

            best_child.parent = None
            node = best_child

    def _augment(self, state, pi, value, training_data, row, col):
        """8-fold augmentation: 4 rotations × 2 reflections."""
        pi_board = pi.reshape(row, col)
        for k in range(4):
            s_rot = np.rot90(state, k, axes=(1, 2))
            p_rot = np.rot90(pi_board, k)
            training_data.append((s_rot.copy(), p_rot.flatten().copy(), value))

            s_flip = np.flip(s_rot, axis=2)
            p_flip = np.fliplr(p_rot)
            training_data.append((s_flip.copy(), p_flip.flatten().copy(), value))
