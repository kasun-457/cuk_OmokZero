"""Entry point for 11x11 Gomoku AlphaZero."""
import argparse
import os

import torch

from config import CFG
from gomoku.gomoku_game import GomokuGame
from neural_net import NeuralNetworkWrapper
from train import Train
from human_play import HumanPlay


def parse_args():
    p = argparse.ArgumentParser(description="11x11 Gomoku AlphaZero")
    p.add_argument("--num_iterations", type=int, default=CFG.num_iterations)
    p.add_argument("--num_games", type=int, default=CFG.num_games)
    p.add_argument("--num_mcts_sims", type=int, default=CFG.num_mcts_sims)
    p.add_argument("--c_puct", type=float, default=CFG.c_puct)
    p.add_argument("--learning_rate", type=float, default=CFG.learning_rate)
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p.add_argument("--resnet_blocks", type=int, default=CFG.resnet_blocks)
    p.add_argument("--model_directory", type=str, default=CFG.model_directory)
    p.add_argument("--num_eval_games", type=int, default=CFG.num_eval_games)
    p.add_argument("--load_model", type=int, default=int(CFG.load_model))
    p.add_argument("--human_play", type=int, default=int(CFG.human_play))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.num_iterations = args.num_iterations
    CFG.num_games = args.num_games
    CFG.num_mcts_sims = args.num_mcts_sims
    CFG.c_puct = args.c_puct
    CFG.learning_rate = args.learning_rate
    CFG.epochs = args.epochs
    CFG.batch_size = args.batch_size
    CFG.resnet_blocks = args.resnet_blocks
    CFG.model_directory = args.model_directory
    CFG.num_eval_games = args.num_eval_games
    CFG.load_model = bool(args.load_model)
    CFG.human_play = bool(args.human_play)

    print("=" * 60)
    if torch.cuda.is_available():
        print(f"[Main] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[Main] CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
    else:
        print("[Main] CUDA not available — running on CPU")
    print("=" * 60)

    game = GomokuGame()
    net = NeuralNetworkWrapper(game)

    best_model_path = os.path.join(CFG.model_directory, "best_model.pt")
    if CFG.load_model and os.path.exists(best_model_path):
        net.load_model("best_model")
    else:
        print("사전 학습 모델 없음. 처음부터 학습을 시작합니다.")

    if CFG.human_play:
        human = HumanPlay(game, net)
        human.play()
    else:
        trainer = Train(game, net)
        trainer.start()
