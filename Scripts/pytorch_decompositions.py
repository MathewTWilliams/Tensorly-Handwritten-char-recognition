# Author: Matt Williams
# Version: 4/13/2022

import tensorly as tl
from tensorly.decomposition import parafac,tucker
from torch.nn import Conv2d


def cp_decomposition_cnn_layer(layer, rank):

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
    # Grouped convolutions 

    