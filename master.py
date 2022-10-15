from torchvision import datasets, transforms
import torch
import time
from utils.trainer import objective
from utils.model import MimoCnnModel
from config import master_config
import random
import optuna
from functools import partial

study_name = str(random.randint(100000, 999999))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
master_config.device = device
o_config = master_config.optuna_config
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])

datasets = {'train_dataset': datasets.MNIST("../data", train=True, download=True, transform=transform),
            'test_dataset': datasets.MNIST("../data", train=False, transform=transform)}

study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=o_config.multivariate,
                                                               group=o_config.grouped,
                                                               n_startup_trials=o_config.n_random_trials),
                            direction='maximize', study_name=study_name)

study.optimize(partial(objective, datasets=datasets,
                       study_name=study_name, mimo_model=MimoCnnModel,
                       config=master_config), n_trials=50)
