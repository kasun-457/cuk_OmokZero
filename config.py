"""Configuration for 11x11 Gomoku AlphaZero."""


class CFG:
    # Self-play & training iterations
    num_iterations = 50
    num_games = 100
    num_mcts_sims = 200
    c_puct = 1.5

    # Optimizer
    l2_val = 1e-4
    momentum = 0.9
    learning_rate = 0.01

    # Temperature schedule
    temp_init = 1.0
    temp_final = 0.01
    temp_thresh = 15   # moves before switching to temp_final

    # Network training
    epochs = 5
    batch_size = 256
    resnet_blocks = 7

    # Dirichlet exploration noise
    dirichlet_alpha = 0.03   # smaller for larger action spaces
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
