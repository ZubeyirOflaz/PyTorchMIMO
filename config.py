from typing import NamedTuple


class optuna_config(NamedTuple):
    # Optuna training parameters
    lr = (5e-6, 5e-1)
    gamma = (0.95, 1)
    optim = 'Adam'
    # Number of randomized trials for TPE Sampler
    class_distribution = 'normal'
    n_random_trials = 35
    # Number of training and validation samples per epoch
    n_train_examples = 5000
    n_test_examples = 2000
    # Enable pruning (custom pruning can be modified from the trainer.py script)
    allow_default_pruning = True
    custom_pruning = True
    # Disable pruning during random trials
    disable_random_trial_pruning = True
    # Custom pruning parameters
    epoch_threshold = 10
    accuracy_threshold = 70
    #TPE Sampler parameters
    multivariate = True
    grouped = True


class training_config(NamedTuple):
    ex = 1


class model_config(NamedTuple):
    num_categories = 10
    final_image_resolution = 6
    num_cnn_layers = 2
    input_image_size = (28, 28)

    hidden_linear_dim = (128, 1024)
    output_linear_dim = (64, 512)
    num_output_channels = (64, 256)

    cnn_dropout = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.50]
    kernel_size = (3,5)
    cnn_channel_base = [4, 8, 16, 32, 64]


class master_config(NamedTuple):
    optuna_config = optuna_config
    model_config = model_config
    training_config = training_config
    num_epochs: int = 50
    num_workers: int = 0
    batch_size = [4, 8]
    ensemble_num = [3,4,5]
