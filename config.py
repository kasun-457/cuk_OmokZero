"""Configuration for 11x11 Gomoku AlphaZero."""


class CFG:
    # Self-play & training iterations
    num_iterations = 80
    num_games = 60                # iter 당 자기대국 수
    num_mcts_sims = 400           # GPU 추론이라 더 많이 가능
    c_puct = 2.0

    # Optimizer
    l2_val = 1e-4
    momentum = 0.9
    learning_rate = 0.01
    lr_milestones = [20, 40, 60]
    lr_gamma = 0.1
    grad_clip = 1.0

    # Temperature schedule
    temp_init = 1.0
    temp_final = 0.01
    temp_thresh = 15

    # Network training (GPU 최적화)
    epochs = 4
    batch_size = 1024             # 3070 Ti VRAM 8GB 활용
    resnet_blocks = 8             # 더 깊은 네트워크
    num_channels = 192            # 채널 확장

    # Replay buffer
    replay_buffer_iters = 5

    # Parallel self-play
    # GPU 워커: 3070 Ti 에 4개 정도가 적당 (워커당 ~1.5GB VRAM)
    num_workers = 4
    worker_device = "cuda"        # 워커도 GPU 사용 (네트워크 큰 경우 필수)

    # Dirichlet exploration noise
    dirichlet_alpha = 0.15        # 11x11 에 맞춤 (이전 0.03 은 너무 적음)
    epsilon = 0.25

    # Model persistence
    model_directory = "./gomoku/models/"
    loss_file = "loss.txt"
    record_loss = True

    # Evaluation
    num_eval_games = 20
    eval_win_rate = 0.55

    # Runtime flags
    load_model = True
    human_play = False
