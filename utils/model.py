import torch
import torch.nn as nn
import pickle
from config import model_config


class MimoCnnModel(nn.Module):
    def __init__(self, trial, ensemble_num: int, cfg: model_config):
        super(MimoCnnModel, self).__init__()
        self.ensemble_num = ensemble_num
        self.hidden_dim = trial.suggest_int('hidden dim', cfg.hidden_linear_dim[0], cfg.hidden_linear_dim[1])
        self.output_dim = trial.suggest_int('output_dim', cfg.output_linear_dim[0],cfg.output_linear_dim[1])
        self.num_channels = trial.suggest_int('num_channels', cfg.num_output_channels[0], cfg.num_output_channels[1])
        self.final_img_resolution = cfg.final_image_resolution
        self.input_dim = self.num_channels * ((self.final_img_resolution - 2) *
                                              ((self.final_img_resolution * ensemble_num) - ((3*ensemble_num)-1)))
        self.conv_module = ConvModule(trial, self.num_channels, self.final_img_resolution, ensemble_num, cfg)
        self.linear_input = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.output_layer = nn.Linear(self.output_dim, cfg.num_categories * ensemble_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = input_tensor.size()[0]
        conv_result = self.conv_module(input_tensor)
        output = self.linear_input(conv_result.reshape(batch_size, -1))
        output = self.hidden_linear(output)
        output = self.output_layer(output)
        output = output.reshape(
            batch_size, self.ensemble_num, -1
        )  # (batch_size, ensemble_num, num_categories)
        output = self.softmax(output)  # (batch_size, ensemble_num, num_categories)
        # print(output.size())
        return output


class ConvModule(nn.Module):
    def __init__(self, trial, num_channels: int, final_img_resolution: int, ensemble_num: int, cfg):
        super(ConvModule, self).__init__()
        layers = []
        num_layers = cfg.num_cnn_layers
        cnn_dropout = trial.suggest_categorical('drop_out_cnn', cfg.cnn_dropout)
        input_channels = 1
        for i in range(num_layers):
            filter_base = cfg.cnn_channel_base
            filter_selections = [y * (i + 1) for y in filter_base]
            num_filters = trial.suggest_categorical(f'num_filters_{i}', filter_selections)
            kernel_size = trial.suggest_int(f'kernel_size_{i}', cfg.kernel_size[0], cfg.kernel_size[1])

            if i < 1:
                pool_stride = 2
            else:
                pool_stride = 1
            layers.append(nn.Conv2d(input_channels, num_filters, (kernel_size,
                                                                  (kernel_size * ensemble_num)), stride=pool_stride))
            if i < 1:
                pool_stride = 2
            else:
                pool_stride = 1
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d((2, 2 * ensemble_num), pool_stride))
                layers.append(nn.Dropout(cnn_dropout))
            input_channels = num_filters
        layers.append(nn.AdaptiveMaxPool2d((cfg.final_image_resolution, cfg.final_image_resolution * ensemble_num)))
        layers.append(nn.Conv2d(input_channels, num_channels, (3, 3 * ensemble_num)))
        self.layers = layers
        self.module = nn.Sequential(*self.layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.module(input_tensor)
        return output
