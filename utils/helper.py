import pandas as pd
import numpy as np
from functools import reduce
from operator import __add__
import logging

log = logging.debug


def create_study_analysis(optuna_study):
    parameters = [i.params for i in optuna_study]
    accuracy = [y.value for y in optuna_study]
    state = [i.state.name for i in optuna_study]
    df = pd.DataFrame(parameters)
    df.insert(0, 'accuracy', accuracy)
    df.assign(trial_state=state, inplace=True)
    df.sort_values('accuracy', ascending=False, inplace=True)
    return df


def weighted_classes_arrhythmia(a_dataset, n_classes=16, return_count=False):
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
    if return_count:
        return weight, count
    else:
        return weight


def calculate_pad(kernel_size):
    conv_padding = reduce(__add__,
                          [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    return conv_padding


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


def determine_stride_padding(input_size, kernel_size, final_resolution):
    stride = 2
    padding = None
    (h, w) = conv2d_output_size(input_size, 0, kernel_size, stride)
    print(f'{h},{w}')
    log('conv executed')
    if h < final_resolution[0] or w < final_resolution[1]:
        stride = 1
        (h, w) = conv2d_output_size(input_size, (0, 0), kernel_size, stride)
        if h < final_resolution[0] or w < final_resolution[1]:
            padding = calculate_pad(kernel_size)
            h, w = input_size[0], input_size[1]
    return stride, padding, (h, w)
