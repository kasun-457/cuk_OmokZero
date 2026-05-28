"""Configuration for 11x11 Gomoku AlphaZero."""


class CFG:
    # Self-play & training iterations
    # 6시간 예산 / RTX 4060 Laptop 8GB 기준으로 약 18~20 iter 도달하도록 튜닝.
    # AlphaZero 는 iter 가 쌓일수록 강해지므로 sims 를 약간 낮춰 처리량을 확보.
    num_iterations = 24           # 도달 못해도 매 iter best_model 저장됨
    num_games = 60                # iter 당 자기대국 수
    num_mcts_sims = 256           # 400→256: 추론이 overhead-bound 라 sims 가 주 속도 레버
    c_puct = 2.0

    # Optimizer
    l2_val = 1e-4
    momentum = 0.9
    learning_rate = 0.01
    # 실제 도달 가능한 iter(~20)에 맞춰 LR 감소 시점 재배치
    lr_milestones = [10, 16, 20]
    lr_gamma = 0.2
    grad_clip = 1.0

    # Temperature schedule
    temp_init = 1.0
    temp_final = 0.01
    temp_thresh = 15

    # Network training (GPU 최적화)
    epochs = 4
    batch_size = 1024             # RTX 4060 Laptop 8GB 활용
    resnet_blocks = 8             # 추론이 overhead-bound 라 이 용량은 사실상 공짜
    num_channels = 192            # 채널 확장

    # Replay buffer
    # iter 수가 제한적이라 데이터 재사용을 늘려 학습 안정화
    replay_buffer_iters = 6

    # Parallel self-play
    # RTX 4060 Laptop: 벤치 결과 6 워커에서 GPU ~97% 포화 (VRAM 여유 큼).
    # batch=1 추론 3ms 의 지연을 워커들이 겹쳐서 메움.
    num_workers = 6
    worker_device = "cuda"        # 워커도 GPU 사용 (네트워크 큰 경우 필수)

    # Dirichlet exploration noise
    dirichlet_alpha = 0.15        # 11x11 에 맞춤 (이전 0.03 은 너무 적음)
    epsilon = 0.25

    # Model persistence
    model_directory = "./gomoku/models/"
    loss_file = "loss.txt"
    record_loss = True

    # Evaluation
    # eval 은 병렬화되지 않아 iter 시간을 크게 먹음 → 20→14 로 축소
    num_eval_games = 14
    eval_win_rate = 0.55

    # Runtime flags
    load_model = True
    human_play = False
