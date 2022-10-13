from typing import NamedTuple


class optuna_config(NamedTuple):
    class_distribution = 'normal'
    # Optuna training parameters

    # Number of randomized trials for TPE Sampler
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


class training_config(NamedTuple):
    ex = 1


class model_config(NamedTuple):
    num_categories = 10


class master_config(NamedTuple):
    optuna_config = optuna_config
    model_config = model_config
    training_config = training_config
    num_epochs: int = 15
    num_workers: int = 0
