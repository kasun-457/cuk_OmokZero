"""Monte Carlo Tree Search with PUCT (AlphaZero-style)."""
import math

import numpy as np

from config import CFG


class TreeNode:
    """A node in the MCTS tree storing statistics for one board state."""

    __slots__ = ("Nsa", "Wsa", "Qsa", "Psa", "action", "children",
                 "child_psas", "parent")

    def __init__(self, parent=None, action=None, psa=0.0):
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.children = []
        self.child_psas = None
        self.parent = parent

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        sqrt_n = math.sqrt(self.Nsa)
        best_uct = -float("inf")
        best = None
        for child in self.children:
            uct = child.Qsa + CFG.c_puct * child.Psa * sqrt_n / (1 + child.Nsa)
            if uct > best_uct:
                best_uct = uct
                best = child
        return best

    def expand(self, valid_mask, psa_vector):
        """Add one child per legal move."""
        self.child_psas = psa_vector.copy()
        action_size = len(valid_mask)
        for idx in range(action_size):
            if valid_mask[idx] > 0:
                self.children.append(TreeNode(parent=self, action=idx,
                                              psa=float(psa_vector[idx])))

    def backup(self, v):
        self.Nsa += 1
        self.Wsa += v
        self.Qsa = self.Wsa / self.Nsa


class MonteCarloTreeSearch:
    """MCTS guided by a policy-value network."""

    def __init__(self, net):
        self.net = net

    def search(self, game, root, temperature):
        """Run CFG.num_mcts_sims simulations and return the chosen child node."""
        for _ in range(CFG.num_mcts_sims):
            node = root
            sim = game.clone()

            # --- Selection ---
            while not node.is_leaf():
                node = node.select_child()
                sim.play_action(node.action)

            game_over, value = sim.check_game_over()

            if not game_over:
                psa_vector, v = self.net.predict(sim.state)
                valid_mask = sim.get_valid_moves()

                # Dirichlet noise at the root for exploration
                if node.parent is None:
                    psa_vector = self._add_dirichlet(psa_vector, valid_mask)

                # Mask illegal moves and renormalise
                psa_vector = psa_vector * valid_mask
                total = psa_vector.sum()
                if total > 0:
                    psa_vector /= total
                else:
                    psa_vector = valid_mask / valid_mask.sum()

                node.expand(valid_mask, psa_vector)
                value = -float(v)   # value is from the perspective of the player
                                    # who just moved, so negate for the node owner
            else:
                value = -float(value)

            # --- Backup ---
            while node is not None:
                node.backup(value)
                value = -value
                node = node.parent

        # --- Select move ---
        if not root.children:
            raise RuntimeError("MCTS: no legal moves available")

        if temperature < 1e-3:
            best = max(root.children, key=lambda c: c.Nsa)
        else:
            visits = np.array([c.Nsa for c in root.children], dtype=np.float64)
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            idx = np.random.choice(len(root.children), p=probs)
            best = root.children[idx]

        return best

    def _add_dirichlet(self, psa_vector, valid_mask):
        n_legal = int(valid_mask.sum())
        if n_legal == 0:
            return psa_vector
        noise = np.zeros_like(psa_vector)
        legal_indices = np.where(valid_mask > 0)[0]
        dir_noise = np.random.dirichlet([CFG.dirichlet_alpha] * n_legal)
        noise[legal_indices] = dir_noise
        return (1 - CFG.epsilon) * psa_vector + CFG.epsilon * noise
