# Author: Matt Williams
# Version: 4/1/2022
# Architecture Reference: 
# https://arxiv.org/pdf/1409.1556.pdf


import torch
from torch.nn import Linear, Conv2d, MaxPool2d, Sequential, Conv
from torch.nn import ReLU, Softmax, CrossEntropyLoss, Dropout
from torch.optim import SGD
from pytorch_utils import *




class VGG11(Py_Torch_Base): 

    def __init__(self, loaders, num_classes): 
        super(VGG11, self).__init__(loaders, num_classes)


    def _define_cnn_layers(self):
        conv_1 = Conv2d(in_channels=1, out_channels=64, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_1)

        conv_2 = Conv2d(in_channels=64, out_channels=128, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_2)

        conv_3 = Conv2d(in_channels=128, out_channels=256, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_3)

        conv_4 = Conv2d(in_channels=256, out_channels=256, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_4)

        conv_5 = Conv2d(in_channels=256, out_channels=512, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_5)

        conv_6 = Conv2d(in_channels=512, out_channels=512, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_6)

        conv_7 = Conv2d(in_channels=512, out_channels=512, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_7)

        conv_8 = Conv2d(in_channels=512, out_channels=512, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_8)


        cnn_layers = Sequential(
            conv_1, 
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            conv_2,
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            conv_3, 
            ReLU(inplace=True),
            conv_4, 
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            conv_5, 
            ReLU(inplace=True),
            conv_6, 
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            conv_7, 
            ReLU(inplace=True),
            conv_8, 
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        return cnn_layers

    def _define_linear_layers(self):
        pass