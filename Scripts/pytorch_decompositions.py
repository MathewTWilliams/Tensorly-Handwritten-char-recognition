# Author: Matt Williams
# Version: 4/13/2022
# references: 
# - https://github.com/JeanKossaifi/tensorly-notebooks/blob/master/05_pytorch_backend/cnn_acceleration_tensorly_and_pytorch.ipynb

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from torch.nn import Conv2d, Sequential
import torch

import warnings
warnings.simplefilter("ignore", UserWarning)


def cp_decomposition_cnn_layer(layer, rank):

    tl.set_backend("pytorch")
    # Perform CP decomposition on weight tensor

    print("----------------------------------------------------------")
    print(layer)
    print("OG Layer Weights:",layer.weight.data.shape)
     
    weights, (last, first, vertical, horizontal) = \
        parafac(layer.weight.data, rank=rank, init='svd')
    print("New Weights:", weights.shape)
    print("----------------------------------------------------------")

    # Perform pointwise convolution to reduce number channels from the 
    # original amount to r number of channels.
    # dilation is the space between the kernel elements
    pointwise_s_to_r_layer = Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride = 1, padding = 0, 
            dilation=layer.dilation, bias = False)

    print("First:", first.shape)
    print(pointwise_s_to_r_layer)
    print("Weight Shape:", pointwise_s_to_r_layer.weight.data.shape)
    print("----------------------------------------------------------")

    # Perform seperable convolutions in the spatial dimensions using grouped convolutions
    depthwise_vertical_layer = Conv2d(in_channels=vertical.shape[1], \
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1), 
            stride = layer.stride, padding = (layer.padding[0], 0), dilation = layer.dilation, #stride = 1
            groups = vertical.shape[1], bias = False)
 
    print("Vertical:", vertical.shape)
    print(depthwise_vertical_layer)
    print("Weight Shape:", depthwise_vertical_layer.weight.data.shape)
    print("----------------------------------------------------------")

    depthwise_horizontal_layer = Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1], kernel_size=(1, horizontal.shape[0]), 
            stride=layer.stride, padding = (0, layer.padding[0]), dilation=layer.dilation, 
            groups = horizontal.shape[1], bias = False)
    
    print("Horizontal:", horizontal.shape)
    print(depthwise_horizontal_layer)
    print("Weight Shape:",depthwise_horizontal_layer.weight.data.shape)
    print("----------------------------------------------------------")
     
    # Perform another pointwise convolution to change the number of channels again
    pointwise_r_to_t_layer = Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1, 
        padding = 0, dilation=layer.dilation, bias = True)

    print("Last:", last.shape)
    print(pointwise_r_to_t_layer)
    print("Weight Shape:", pointwise_r_to_t_layer.weight.data.shape)
    print("----------------------------------------------------------")

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

    return new_layers

def tucker_decomposition_cnn_layer(layer, ranks):

    tl.set_backend("pytorch")
    core, [last,first] = \
        partial_tucker(layer.weight.data, modes = [0,1], rank = ranks, init='svd')

    
    print("----------------------------------------------------------")
    print(layer)
    print("OG Layer Weights:", layer.weight.data.shape)

    # Pointwise convolution that reduces the channels
    first_layer = Conv2d(in_channels = first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, 
            stride = 1, padding = 0, dilation=layer.dilation, bias = False)

    print("----------------------------------------------------------")
    print("first:", first.shape)
    print(first_layer)
    print("Weight Shape:", first_layer.weight.data.shape)

    # A regular 2D convolution layer with core 
    core_layer = Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size, 
            stride = layer.stride, padding = layer.padding,
            dilation = layer.dilation, bias = False)

    print("core:", core.shape)
    print(core_layer)
    print("Weight Shape:", core_layer.weight.data.shape)

    # A pointwise convolution that increase the number of channels
    last_layer = Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride = 1, 
            padding = 0, dilation = layer.dilation, bias = True)

    print("last:", last.shape)
    print(last_layer)
    print("Weight Shape:", last_layer.weight.data.shape)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]

    return new_layers