'''Contains the functions used to create the dataloaders for training and validation'''

import torch
from torch.utils.data import ConcatDataset, DataLoader
from functools import partial

'''collating function for the training dataloader'''

'''Preload data to GPU using collate functions'''

def collate_training(batch, batch_size, ensemble_num):
    # get x and y
    data = [i[0] for i in batch]
    target = [i[1] for i in batch]
    # separate data to ensembles
    batch_range = list(range(0, batch_size))
    ensembles = [data[i * ensemble_num:(i * ensemble_num) + ensemble_num] for i in batch_range]
    ensemble_targets = [target[i * ensemble_num:(i * ensemble_num) + ensemble_num] for i in batch_range]
    # Concatenate tensors to the ensembles
    ensembles = torch.stack([torch.cat(i, dim=2) for i in ensembles])
    ensemble_targets = torch.tensor(ensemble_targets)
    return [ensembles, ensemble_targets]


'''collating function for the test dataloader'''


def collate_test(batch, ensemble_num):
    # get x and y
    data = [i[0] for i in batch]
    target = [i[1] for i in batch]
    # multiply the data and concat as one input
    data_mimo = torch.stack([torch.cat([i] * ensemble_num, dim=2) for i in data])
    return [data_mimo, target]


def create_train_dataloader(dataset, batch_size, ensemble_num, **params):
    '''complete_dataset = [[dataset] * ensemble_num]
    complete_dataset = ConcatDataset(complete_dataset)'''
    train_loader = DataLoader(dataset, batch_size=(batch_size * ensemble_num),
                              collate_fn=partial(collate_training,
                                                 batch_size=batch_size, ensemble_num=ensemble_num),
                              **params)
    return train_loader


def create_test_dataloader(dataset, batch_size, ensemble_num, **params):
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=partial(collate_test,
                                                ensemble_num=ensemble_num),
                             **params)
    return test_loader


if __name__ == '__main__':
    batch_size = 8
    ensemble_num = 3
    test_tensor = [torch.randn(1, 128, 128), torch.tensor(1)]
    test_batch = [[test_tensor] * (batch_size * ensemble_num)][0]
    test_func = collate_training(test_batch, batch_size, ensemble_num)
    test_func_test = collate_test(test_batch, ensemble_num)
