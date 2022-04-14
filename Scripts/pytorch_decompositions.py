# Author: Matt Williams
# Version: 4/13/2022
# references: 
# - https://github.com/JeanKossaifi/tensorly-notebooks/blob/master/05_pytorch_backend/cnn_acceleration_tensorly_and_pytorch.ipynb

import tensorly as tl
from tensorly.decomposition import parafac,tucker
from torch.nn import Conv2d, Sequential
import torch


def cp_decomposition_cnn_layer(layer, rank):

    tl.set_backend("pytorch")
    # Perform CP decomposition on weight tensor
    last, first, vertical, horizontal = \
        parafac(layer.weight.data, rank=rank, init='svd')

    # Perform pointwise convolution to reduce number channels from the 
    # original amount to r number of channels.

    # dilation is the space between the kernel elements
    pointwise_s_to_r_layer = Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride = 1, padding = 0, 
            dilation=layer.dilation, bias = False)


    # Perform seperable convolutions in the spatial dimensions using grouped convolutions
    depthwise_vertical_layer = Conv2d(in_channels=vertical.shape[1], \
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1), 
            stride = layer.stride, padding = (layer.padding[0], 0), dilation = layer.dilation, #stride = 1
            groups = vertical.shape[1], bias = False)


    depthwise_horizontal_layer = Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1], kernel_size=(1, horizontal.shape[0]), 
            stride=layer.stride, padding = (0, layer.padding[0]), dilation=layer.dilation, 
            groups = horizontal.shape[1], bias = False)
    

    # Perform another pointwise convolution to change the number of channels again

    pointwise_r_to_t_layer = Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1, 
        padding = 0, dilation=layer.dilation, bias = True)


    pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return Sequential(*new_layers)