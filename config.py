"""Configuration for 11x11 Gomoku AlphaZero."""


class CFG:
    # Self-play & training iterations
    # 6시간 예산 / RTX 4060 Laptop 8GB 기준으로 약 18~20 iter 도달하도록 튜닝.
    # AlphaZero 는 iter 가 쌓일수록 강해지므로 sims 를 약간 낮춰 처리량을 확보.
    num_iterations = 24           # 도달 못해도 매 iter current/best 저장됨
    num_games = 60                # iter 당 자기대국 수
    # 실측: iter당 self-play ~42분(초반 무작위라 게임이 거의 121수까지 김).
    # sims 를 낮춰 throughput 을 올려 더 많은 iter(학습 누적)를 확보.
    num_mcts_sims = 160           # 256→160
    c_puct = 2.0

    # Optimizer
    l2_val = 1e-4
    momentum = 0.9
    learning_rate = 0.01
    # 실제 도달 가능한 iter(~12)에 맞춰 LR 감소 시점 재배치
    lr_milestones = [6, 10, 13]
    lr_gamma = 0.3
    grad_clip = 1.0

    # Temperature schedule
    temp_init = 1.0
    temp_final = 0.01
    temp_thresh = 15

    # Network training (GPU 최적화)
    # self-play 가 iter 시간의 대부분이라 epoch 를 늘려도 비용은 미미.
    # 누적 버퍼에서 더 충분히 학습해 정책이 균일분포를 빨리 벗어나게 함.
    epochs = 6
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
    # eval 은 병렬화되지 않아 iter 시간을 먹음. 이제 게이트가 아니라
    # 진척도 로깅/best 추적 용도이므로 8게임으로 축소.
    num_eval_games = 8
    eval_win_rate = 0.55          # 미사용(참고용). 채택 기준은 train.py 의 0.5

    # Runtime flags
    load_model = True
    human_play = False
