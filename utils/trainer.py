import torch
import torch.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pickle
import optuna
from optuna.trial import TrialState
from dataloaders import create_train_dataloader, create_test_dataloader
from config import config

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

'''Custom Optuna pruner that prunes a trial if a minimum accuracy stated is not reached by the given epoch threshold'''
def pruner(epoch_threshold, current_epoch, min_acc, current_acc):
    if current_epoch > epoch_threshold and current_acc < min_acc:
        return True
    else:
        return False




def objective(trial, mimo_model, datasets, study_name, config = config):
    # Model and main parameter initialization
    num_epochs = 50
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    ensemble_num = trial.suggest_categorical('ensemble_num', [3, 4])
    train_loader = create_train_dataloader(datasets['train_dataset'], batch_size, ensemble_num)
    test_loader = create_test_dataloader(datasets['test_dataset'], batch_size, ensemble_num)
    try:
        model = mimo_model(trial=trial).to(device)
        print(model)
    except Exception as e:
        print('Infeasible model, trial will be skipped')
        print(e)
        raise optuna.exceptions.TrialPruned()
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 5e-6, 5e-1, log=True)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.95, 1)
    scheduler = StepLR(optimizer, step_size=(len(train_loader[0])), gamma=gamma)

    # Training and eval loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for datum in zip(*train_loader):
            model_inputs = torch.cat([data[0] for data in datum], dim=3).to(device)
            targets = torch.stack([data[1] for data in datum]).to(device)
            optimizer.zero_grad()
            outputs = model(model_inputs)
            ensemble_num, batch_size = list(targets.size())

            loss = F.nll_loss(
                outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
            )
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'{epoch}: {train_loss}')
        model.eval()
        test_loss = 0
        test_size = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                model_inputs = torch.cat([data[0]] * ensemble_num, dim=3).to(device)
                target = data[1].to(device)
                outputs = model(model_inputs)
                output = torch.mean(outputs, axis=1)
                test_size += len(target)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= test_size
        acc = 100.0 * correct / test_size
        print(f'epoch {epoch}: {acc}')
        trial.report(acc, epoch)
        should_prune = False
        # Check if it should be pruned by the default pruner
        should_prune = config.allow_default_pruning and trial.should_prune()
        # Check if it should not be pruned during random trials
        should_prune != config.disable_random_trial_pruning and trial.number > config.n_random_trials
        # Check if custom pruner should be used
        if config.custom_pruning:
            should_prune = should_prune or pruner(config.epoch_threshold,epoch,config.accuracy_threshold,acc)
        if should_prune:
            raise optuna.exceptions.TrialPruned()
    torch.save(model.state_dict(), f"model_repo\\{trial.number}_{study_name}.pyt")

    '''with open("model_repo\\{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)'''

    return acc
