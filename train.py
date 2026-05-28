"""Self-play training loop for 11x11 Gomoku AlphaZero (parallel self-play)."""
import io
import os
import multiprocessing as mp
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch

from config import CFG
from mcts import MonteCarloTreeSearch, TreeNode
from neural_net import NeuralNetworkWrapper
from evaluate import Evaluate
from gomoku.gomoku_game import GomokuGame


# ====================================================================
#  Worker process functions  (모듈 최상위에 있어야 spawn 으로 pickle 가능)
# ====================================================================

def _augment(state, pi, value, training_data, row, col):
    """8-fold augmentation: 4 rotations x 2 reflections."""
    pi_board = pi.reshape(row, col)
    for k in range(4):
        s_rot = np.rot90(state, k, axes=(1, 2))
        p_rot = np.rot90(pi_board, k)
        training_data.append((s_rot.copy(), p_rot.flatten().copy(), value))
        s_flip = np.flip(s_rot, axis=2)
        p_flip = np.fliplr(p_rot)
        training_data.append((s_flip.copy(), p_flip.flatten().copy(), value))


def _play_one_game(net):
    """단일 자기대국 실행. (states, pis, values) 리스트(증강 포함) 반환."""
    game = GomokuGame()
    mcts = MonteCarloTreeSearch(net)
    node = TreeNode()
    self_play_data = []
    move_count = 0
    training_data = []

    while True:
        temp = CFG.temp_init if move_count < CFG.temp_thresh else CFG.temp_final
        best_child = mcts.search(game, node, temp)

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
            for i, (s, p, _) in enumerate(reversed(self_play_data)):
                v = value * ((-1) ** i)
                _augment(s, p, v, training_data, game.row, game.column)
            return training_data

        best_child.parent = None
        node = best_child


def _worker_play_games(args):
    """
    Worker 프로세스 진입점.
    각 워커는 자체 네트워크를 GPU(또는 CPU)에 올리고 자기대국을 돌린다.
    """
    state_dict_bytes, num_games, seed, worker_id, device = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU 메모리 효율을 위해 일부 PyTorch 설정 조정
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    game = GomokuGame()
    net = NeuralNetworkWrapper(game, device=device)

    # state_dict 복원 — 항상 CPU 에 먼저 로드한 뒤 디바이스로 이동
    buf = io.BytesIO(state_dict_bytes)
    state_dict = torch.load(buf, map_location="cpu")
    net.net.load_state_dict(state_dict)
    net.net.to(net.device)
    net.net.eval()

    data = []
    for i in range(num_games):
        data.extend(_play_one_game(net))
        print(f"  [worker {worker_id} | {device}] game {i+1}/{num_games} done",
              flush=True)
    return data


# ====================================================================
#  Trainer
# ====================================================================

class Train:
    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.eval_net = NeuralNetworkWrapper(game)

        # 워커 수 결정
        cores = os.cpu_count() or 2
        if CFG.num_workers > 0:
            self.num_workers = CFG.num_workers
        else:
            self.num_workers = max(1, min(12, cores - 1))

        # 워커 디바이스 결정
        self.worker_device = getattr(CFG, "worker_device", "cpu")
        if self.worker_device == "cuda" and not torch.cuda.is_available():
            print("[Train] CUDA unavailable for workers — using CPU")
            self.worker_device = "cpu"

        print(f"[Train] Using {self.num_workers} parallel workers on "
              f"'{self.worker_device}' (detected {cores} CPU cores)")
        if torch.cuda.is_available():
            print(f"[Train] GPU: {torch.cuda.get_device_name(0)} | "
                  f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

        # Replay buffer: 최근 N 이터레이션의 자기대국 데이터 보존
        buf_iters = getattr(CFG, "replay_buffer_iters", 5)
        self.replay_buffer = deque(maxlen=buf_iters)
        print(f"[Train] Replay buffer: keeping last {buf_iters} iterations")

    # ----------------------------------------------------------------

    def start(self):
        # 평가 기준선(직전 iteration 모델)을 위한 시드 저장.
        # 초기 가중치를 current_model / best_model 로 한 번 기록해 둔다.
        self.net.save_model()
        self.net.save_model("best_model")

        for iteration in range(1, CFG.num_iterations + 1):
            print(f"\n=== Iteration {iteration}/{CFG.num_iterations} ===")

            t0 = time.time()
            new_data = self._parallel_self_play()
            t_sp = time.time() - t0
            print(f"  Self-play: {len(new_data)} samples "
                  f"in {t_sp:.1f}s ({CFG.num_games/t_sp:.2f} games/s)")

            # Replay buffer 갱신: 새 데이터 추가, 오래된 건 자동 폐기
            self.replay_buffer.append(new_data)
            training_data = []
            for chunk in self.replay_buffer:
                training_data.extend(chunk)
            print(f"  Replay buffer: {len(training_data)} total samples "
                  f"({len(self.replay_buffer)} iterations stored)")

            # 평가 기준선 = 직전 iteration 에서 채택된 모델(current_model)
            self.eval_net.load_model()

            # 학습
            self.net.train(training_data)
            self.net.step_scheduler()

            # === 항상 학습 결과를 채택(누적). 절대 되돌리지 않는다. ===
            # (구버전은 eval 승률 < 0.55 면 학습을 통째로 폐기해 정책이
            #  균일분포에서 못 벗어났음. 현대 AlphaZero 방식으로 변경.)
            self.net.save_model()

            # 평가는 '진척도 로깅 + best 추적' 용도로만 사용 (게이트 아님)
            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)
            evaluator = Evaluate(current_mcts, eval_mcts, self.game)
            wins, losses = evaluator.evaluate()

            played = wins + losses
            win_rate = wins / played if played > 0 else 0.0
            print(f"Eval  wins={wins}  losses={losses}  win_rate={win_rate:.2f}")

            # 직전 모델보다 약하지 않으면 best_model 갱신 (학습은 이미 채택됨)
            if win_rate >= 0.5:
                print("New best model saved.")
                self.net.save_model("best_model")
            else:
                print("best_model 유지 (단, 학습 결과는 current_model 로 누적됨).")

    # ----------------------------------------------------------------

    def _parallel_self_play(self):
        """현재 네트워크를 워커들에게 배포하고 자기대국 병렬 실행."""
        # 1) 현재 네트워크 가중치를 bytes 로 직렬화
        buf = io.BytesIO()
        torch.save(self.net.net.state_dict(), buf)
        state_bytes = buf.getvalue()

        # 2) 게임 수를 워커별로 분배
        per_worker = [CFG.num_games // self.num_workers] * self.num_workers
        for i in range(CFG.num_games % self.num_workers):
            per_worker[i] += 1

        args_list = [
            (state_bytes, n, int(np.random.randint(0, 2**31 - 1)), wid,
             self.worker_device)
            for wid, n in enumerate(per_worker) if n > 0
        ]

        # 3) Windows 호환을 위해 spawn 컨텍스트 사용
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(args_list)) as pool:
            results = pool.map(_worker_play_games, args_list)

        training_data = []
        for r in results:
            training_data.extend(r)
        return training_data
