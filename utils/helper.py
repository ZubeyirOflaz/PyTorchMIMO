import pandas as pd
import numpy as np
from functools import reduce
from operator import __add__
import logging
import time
import torch
from torch.utils.data import DataLoader
import thop
import torchmetrics
import torch.distributions as dists
import os
import json

log = logging.debug

'''Function to create a pandas dataframe from trials. Inputs an Optuna study, outputs a Pandas dataframe'''


def create_study_analysis(optuna_study,metrics_dict):
    parameters = [i.params for i in optuna_study]
    accuracy = [y.value for y in optuna_study]
    state = [i.state.name for i in optuna_study]
    df = pd.DataFrame(parameters)
    metrics_df = pd.DataFrame(metrics_dict['metric_list'])
    df = pd.concat([df,metrics_df], axis=1)
    df.insert(0, 'accuracy', accuracy)
    df.assign(trial_state=state, inplace=True)
    df.sort_values('accuracy', ascending=False, inplace=True)
    return df


'''Function used to find the class distribution of a dataset
Inputs the target dataset, number of classes within the dataset. 
Outputs the weight per class.
'''


def weighted_classes(a_dataset, n_classes: int):
    count = [0] * n_classes
    for _, y in enumerate(a_dataset):
        count[y[1].numpy()[0]] += 1
    weight_per_class = [0.] * n_classes
    n = float(sum(count))
    for i in range(n_classes):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = n / float(count[i])
    weight = [0] * len(a_dataset)
    for idx, val in enumerate(a_dataset):
        weight[idx] = weight_per_class[val[1]]
    return weight


'''Function for calculating same padding in PyTorch
Inputs the kernel size that will be applied
Outputs the padding size'''


def calculate_pad(kernel_size):
    conv_padding = reduce(__add__,
                          [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    return conv_padding


'''Function for calculating the tensor size after and convolution layer is applied
Inputs the size of input image, the size of padding kernel size and stride of a convolution or pooling layer
Outputs the size of resulting tensor'''


def conv2d_output_size(input_size, padding, kernel_size, stride):
    if isinstance(padding, int):
        padding = (padding,) * 2
    if isinstance(stride, int):
        stride = (stride,) * 2

    output_size = (
        np.floor((input_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int)
    )
    log(output_size)
    return output_size


'''Function used to regulate the tensor size for Optuna models. It reduces stride and introduces padding when needed
Inputs the size of the input tensor, kernel size of a conv or pooling layer and final allowed resolution for the model
Outputs the stride amount and, if needed, padding size'''

def determine_stride_padding(input_size, kernel_size, final_resolution):
    stride = 2
    padding = None
    (h, w) = conv2d_output_size(input_size, 0, kernel_size, stride)
    log('conv executed')
    if h < final_resolution[0] or w < final_resolution[1]:
        stride = 1
        (h, w) = conv2d_output_size(input_size, (0, 0), kernel_size, stride)
        if h < final_resolution[0] or w < final_resolution[1]:
            padding = calculate_pad(kernel_size)
            h, w = input_size[0], input_size[1]
    return stride, padding, (h, w)


def get_runtime_model_size(dataloader: DataLoader, model, batch_size: int):
    dataset_len = len(dataloader.dataset)
    preds = []
    targets = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    start_time = time.time()
    for data in dataloader:
        outputs = model(data[0].to(device))
        output = torch.mean(outputs, axis=1).cpu().detach()
        preds.append(output.cpu().detach())
        targets.append(data[1].cpu().detach())
    runtime = time.time() - start_time
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    flops, params = thop.profile(model.to(device), inputs=(data[0].to(device),), verbose=False)
    flops = flops / batch_size
    runtime = runtime / dataset_len
    metrics = {'flops': flops,
               'params': params,
               'runtime': runtime}
    results = {'preds': preds,
               'targets': targets}
    return metrics, results


def get_metrics(predictions, targets, ece_bins=10, n_class=2):
    metrics = dict()
    metrics['nll'] = -dists.Categorical(predictions).log_prob(targets).mean().item()
    metrics['ece'] = torchmetrics.functional.calibration_error(predictions, targets, n_bins=ece_bins).item()
    metrics['mce'] = torchmetrics.functional.calibration_error(predictions, targets, n_bins=ece_bins).item()
    metrics['auroc'] = torchmetrics.functional.auroc(predictions, targets, num_classes=n_class).item()
    return metrics
'''Function to record additional performance metrics on the models and save them during ongoing trials
Inputs path and metrics to be recorded
Outputs a JSON file in the specified path'''

def record_metrics(metrics, path):
    if not os.path.isfile(path):
        with open(path, 'w') as fout:
            metric={'metric_list':[metrics]}
            json.dump(metric,fout)
    else:
        with open(path,'r+') as fin:
            metric = json.load(fin)
            metric['metric_list'].append(metrics)
            fin.seek(0)
            # convert back to json.
            json.dump(metric, fin, indent=4)
    return None
def read_metrics(path):
    with open(path, 'r+') as fin:
        metrics = json.load(fin)
        return metrics


if __name__ == "__main__":
    a = 1


    def test(val):
        global a
        a = val


    test(2)
    print(a)
