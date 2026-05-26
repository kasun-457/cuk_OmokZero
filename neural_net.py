"""Policy-Value ResNet for 11x11 Gomoku (PyTorch)."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import CFG


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class GomokuNet(nn.Module):
    """Dual-head ResNet: shared tower → policy head + value head.

    Input: (batch, 4, 11, 11)  — 4 feature planes
    Policy output: (batch, 11*11)  softmax probabilities
    Value output:  (batch,)        tanh scalar in [-1, 1]
    """

    def __init__(self, board_size, action_size, num_channels=128, num_blocks=7):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # Convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(4, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.tower(self.stem(x))

        p = self.policy_conv(x).flatten(1)
        p = F.softmax(self.policy_fc(p), dim=1)

        v = self.value_conv(x).flatten(1)
        v = self.value_fc(v).squeeze(1)

        return p, v


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class NeuralNetworkWrapper:
    def __init__(self, game):
        self.board_size = game.row
        self.action_size = game.action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = GomokuNet(
            board_size=self.board_size,
            action_size=self.action_size,
            num_channels=128,
            num_blocks=CFG.resnet_blocks,
        ).to(self.device)

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=CFG.learning_rate,
            momentum=CFG.momentum,
            weight_decay=CFG.l2_val,
        )

    # ------------------------------------------------------------------

    def predict(self, state):
        """state: numpy (4, 11, 11) → (policy_np, value_float)."""
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            p, v = self.net(t)
        return p[0].cpu().numpy(), float(v[0].cpu())

    def train(self, training_data):
        """training_data: list of (state, pi, v)."""
        print(f"\nTraining on {len(training_data)} samples.\n")
        self.net.train()

        for epoch in range(CFG.epochs):
            np.random.shuffle(training_data)
            total_pi_loss = 0.0
            total_v_loss = 0.0
            batches = 0

            for i in range(0, len(training_data), CFG.batch_size):
                batch = training_data[i:i + CFG.batch_size]
                states, pis, vs = zip(*batch)

                s = torch.tensor(np.array(states), dtype=torch.float32,
                                 device=self.device)
                pi_target = torch.tensor(np.array(pis), dtype=torch.float32,
                                         device=self.device)
                v_target = torch.tensor(np.array(vs), dtype=torch.float32,
                                        device=self.device)

                self.optimizer.zero_grad()
                p_pred, v_pred = self.net(s)

                pi_loss = -(pi_target * (p_pred + 1e-8).log()).sum(dim=1).mean()
                v_loss = F.mse_loss(v_pred, v_target)
                loss = pi_loss + v_loss
                loss.backward()
                self.optimizer.step()

                total_pi_loss += pi_loss.item()
                total_v_loss += v_loss.item()
                batches += 1

            print(f"  Epoch {epoch+1}/{CFG.epochs}  "
                  f"pi_loss={total_pi_loss/batches:.4f}  "
                  f"v_loss={total_v_loss/batches:.4f}")

            if CFG.record_loss:
                os.makedirs(CFG.model_directory, exist_ok=True)
                with open(CFG.model_directory + CFG.loss_file, "a") as f:
                    f.write(f"{total_pi_loss/batches:.6f}|{total_v_loss/batches:.6f}\n")

        print()

    # ------------------------------------------------------------------

    def save_model(self, filename="current_model"):
        os.makedirs(CFG.model_directory, exist_ok=True)
        path = os.path.join(CFG.model_directory, filename + ".pt")
        torch.save({
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"Model saved: {path}")

    def load_model(self, filename="current_model"):
        path = os.path.join(CFG.model_directory, filename + ".pt")
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Model loaded: {path}")
