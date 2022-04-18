# Author: Matt Williams
# Version 3/22/2022
# Architecture Reference: 
# https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/

import torch
from torch.nn import Linear, Conv2d, MaxPool2d, Sequential
from torch.nn import ReLU, Softmax, CrossEntropyLoss, Dropout
from torch.optim import SGD
from pytorch_utils import Py_Torch_Base, initialize_weights_bias, decompose_cnn_layers
from pytorch_decompositions import tucker_decomposition_cnn_layer, cp_decomposition_cnn_layer
from constants import Decomposition

class AlexNet(Py_Torch_Base):

    def __init__(self, loaders, num_classes): 
        super(AlexNet, self).__init__(loaders, num_classes)

    def _define_cnn_layers(self):
        
        conv_1 = Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding = 0, stride = 1) 
        initialize_weights_bias(conv_1)

        conv_2 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding = 2, stride = 1)
        initialize_weights_bias(conv_2)

        conv_3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1, stride = 1)
        initialize_weights_bias(conv_3)

        conv_4 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 1, stride = 1)
        initialize_weights_bias(conv_4)

        conv_5 = Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding = 1, stride = 1)
        initialize_weights_bias(conv_5)
        

        cnn_layers = Sequential(
            conv_1, # 24 x 24 x 32
            ReLU(inplace = True),
            MaxPool2d(kernel_size=2, stride = 2), # 12 x 12 x 32 
            conv_2, # 12 x 12 x 64
            ReLU(inplace = True),
            MaxPool2d(kernel_size=2, stride=2), # 6 x 6 x 64
            conv_3, # 6 x 6 x 128
            ReLU(inplace=True), 
            conv_4, # 6 x 6 x 128
            ReLU(inplace=True), 
            conv_5, # 6 x 6 x 64
            ReLU(inplace=True), 
            MaxPool2d(kernel_size=2, stride = 2), # 3 x 3 x 64  
            Dropout(p = 0.3),
        )

        return cnn_layers

    def _define_linear_layers(self):
        linear_1 = Linear(576, out_features=256)
        initialize_weights_bias(linear_1)


        linear_2 = Linear(256, out_features=64)
        initialize_weights_bias(linear_2)


        linear_3 = Linear(64, out_features=self._num_classes)
        initialize_weights_bias(linear_3)

        linear_layers = Sequential(
            linear_1, 
            ReLU(inplace = True),
            Dropout(p = 0.3),
            linear_2,
            ReLU(inplace=True), 
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


class AlexNet_Decomposed(AlexNet): 

    def __init__(self, loaders, num_classes, decomposition = Decomposition.CP):
        self._decomposition = decomposition 
        super(AlexNet_Decomposed, self).__init__(loaders, num_classes)

    def _define_cnn_layers(self):
        org_cnn_layers = super(AlexNet_Decomposed,self)._define_cnn_layers()
        return decompose_cnn_layers(org_cnn_layers, self._decomposition)