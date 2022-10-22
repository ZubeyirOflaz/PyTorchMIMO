from torchvision import datasets, transforms
import torch
import time
from utils.trainer import objective
from utils.model import MimoCnnModel
from config import master_config
import random
import optuna
from functools import partial
import pickle
from utils.helper import create_study_analysis
from optuna.trial import TrialState

study_name = str(random.randint(100000, 999999))
study_path = None
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
master_config.device = device
o_config = master_config.optuna_config
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])

datasets = {'train_dataset': datasets.MNIST("../data", train=True, download=True, transform=transform),
            'test_dataset': datasets.MNIST("../data", train=False, transform=transform)}

if study_path:
    with open(study_path, 'rb') as fin:
        study = pickle.load(fin)
else:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=o_config.multivariate,
                                                                   group=o_config.grouped,
                                                                   n_startup_trials=o_config.n_random_trials),
                                direction='maximize', study_name=study_name)


    study.optimize(partial(objective, datasets=datasets,
                       study_name=study_name, mimo_model=MimoCnnModel,
                       config=master_config), n_trials=master_config.num_trials)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    with open(f"model_repo\\{study.best_trial.number}_{study_name}.pkl", "rb") as fin:
        best_model = pickle.load(fin)
    with open(f"model_repo\\best_models\\{study.best_trial.value}_{study_name}.pkl", "wb") as fout:
        pickle.dump(best_model, fout)

    trial_dataframe = create_study_analysis(study.get_trials(deepcopy=True))
    with open(f'model_repo\\study_{study.study_name}.pkl', 'wb') as fout:
        pickle.dump(study, fout)
