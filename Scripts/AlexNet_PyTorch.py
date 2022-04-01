# Author: Matt Williams
# Version 3/22/2022
# Architecture Reference: 
# https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/

import torch
from torch.nn import Linear, Conv2d, MaxPool2d, Sequential
from torch.nn import ReLU, Softmax, CrossEntropyLoss, Dropout
from torch.optim import SGD
from pytorch_utils import Py_Torch_Base


class AlexNet(Py_Torch_Base):

    def __init__(self, loaders, num_classes): 
        super(AlexNet, self).__init__(loaders, num_classes)

    def _define_cnn_layers(self):
        
        conv_1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding = 0, stride = 2)
        torch.nn.init.xavier_uniform_(conv_1.weight)
        if conv_1.bias is not None:
            torch.nn.init.zeros_(conv_1.bias)

        conv_2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding = 2, stride = 1)
        torch.nn.init.xavier_uniform_(conv_2.weight)
        if conv_2.bias is not None:
            torch.nn.init.zeros_(conv_2.bias)

        conv_3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1, stride = 1)
        torch.nn.init.xavier_uniform_(conv_3.weight)
        if conv_3.bias is not None:
            torch.nn.init.zeros_(conv_3.bias)

        conv_4 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1, stride = 1)
        torch.nn.init.xavier_uniform_(conv_4.weight)
        if conv_4.bias is not None:
            torch.nn.init.zeros_(conv_4.bias)
        
        conv_5 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1, stride = 1)
        torch.nn.init.xavier_uniform_(conv_5.weight)
        if conv_5.bias is not None:
            torch.nn.init.zeros_(conv_5.bias)
        

        cnn_layers = Sequential(
            conv_1, 
            ReLU(inplace = True),
            MaxPool2d(kernel_size=3, stride = 1), 
            conv_2, 
            ReLU(inplace = True),
            MaxPool2d(kernel_size=3, stride=1), 
            conv_3,
            conv_4, 
            conv_5,
            MaxPool2d(kernel_size=3, stride = 2),
            ReLU(inplace = True),
            Dropout(p = 0.3)
        )

        return cnn_layers

    def _define_linear_layers(self):
        linear_1 = Linear(3200, out_features=1600)
        torch.nn.init.xavier_uniform_(linear_1.weight)
        if linear_1.bias is not None: 
            torch.nn.init.zeros_(linear_1.bias)


        linear_2 = Linear(1600, out_features=100)
        torch.nn.init.xavier_uniform_(linear_2.weight)
        if linear_2.bias is not None: 
            torch.nn.init.zeros_(linear_2.bias)


        linear_3 = Linear(100, out_features=self._num_classes)
        torch.nn.init.xavier_uniform_(linear_3.weight)
        if linear_3.bias is not None: 
            torch.nn.init.zeros_(linear_3.bias)

        linear_layers = Sequential(
            linear_1, 
            ReLU(inplace = True),
            Dropout(p = 0.3),
            linear_2, 
            linear_3,

            # No Softmax layer at the end
            # Py-Torch's implementation of Cross Entropy Loss usings LogSoftmax
            # with negative log likelihood loss
            #Softmax (dim = -1)
        )

        return linear_layers


    def _define_optimizer(self):
        return SGD(self.parameters(), lr = 0.01, momentum=0.9)


    def _define_loss_function(self):
        return CrossEntropyLoss()
        

