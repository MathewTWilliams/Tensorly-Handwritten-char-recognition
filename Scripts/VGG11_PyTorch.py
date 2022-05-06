# Author: Matt Williams
# Version: 4/1/2022
# Architecture Reference: 
# https://arxiv.org/pdf/1409.1556.pdf


import torch
from torch.nn import Linear, Conv2d, MaxPool2d, Sequential
from torch.nn import ReLU, Softmax, CrossEntropyLoss, Dropout
from torch.optim import SGD
from pytorch_utils import *
from pytorch_decompositions import decompose_cnn_layers
from constants import Decomposition


class VGG11(Py_Torch_Base): 
    """Defines the Py-Torch implementation of the scaled Down VGG-11 model"""
    def __init__(self, loaders, num_classes): 
        super(VGG11, self).__init__(loaders, num_classes)


    def _define_cnn_layers(self):
        """Defines the CNN layers of the model"""
        conv_1 = Conv2d(in_channels=1, out_channels=32, kernel_size= 3, padding = 1, stride = 1) 
        initialize_weights_bias(conv_1)

        conv_2 = Conv2d(in_channels=32, out_channels=64, kernel_size= 3, padding = 1, stride = 1) 
        initialize_weights_bias(conv_2)

        conv_3 = Conv2d(in_channels=64, out_channels=128, kernel_size= 3, padding = 1, stride = 1) 
        initialize_weights_bias(conv_3)

        conv_4 = Conv2d(in_channels=128, out_channels=128, kernel_size= 3, padding = 1, stride = 1) 
        initialize_weights_bias(conv_4)

        conv_5 = Conv2d(in_channels=128, out_channels=256, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_5)

        conv_6 = Conv2d(in_channels=256, out_channels=256, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_6)

        conv_7 = Conv2d(in_channels=256, out_channels=256, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_7)

        conv_8 = Conv2d(in_channels=256, out_channels=256, kernel_size= 3, padding = 1, stride = 1)
        initialize_weights_bias(conv_8)


        cnn_layers = Sequential(
            conv_1, # 28 x 28 x 32
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2), # 14 x 14 x 32
            conv_2, # 14 x 14 x 64
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding = 1), # 8 x 8 x 64
            conv_3, # 8 x 8 x 128            
            ReLU(inplace=True),
            conv_4, # 8 x 8 x 128
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2), # 4 x 4 x 128
            conv_5, # 4 x 4 x 256
            ReLU(inplace=True),
            conv_6, # 4 x 4 x 256
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2), # 2 x 2 x 256
            conv_7, # 2 x 2 x 256
            ReLU(inplace=True),
            conv_8, # 2 x 2 x 256
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2), #1 x 1 x 256
        )

        return cnn_layers

    def _define_linear_layers(self):
        """Defines the Linear layers of the model"""
        linear_1 = Linear(in_features=256, out_features=128)
        initialize_weights_bias(linear_1)
        
        linear_2 = Linear(in_features=128, out_features=64) 
        initialize_weights_bias(linear_2)

        linear_3 = Linear(in_features=64, out_features=self._num_classes)
        initialize_weights_bias(linear_3)
    
        linear_layers = Sequential(
            linear_1,
            ReLU(inplace = True),
            Dropout(p = 0.3), 
            linear_2,
            ReLU(inplace = True),
            Dropout(p=0.3), 
            linear_3,
            ReLU(inplace = True),
            # No Softmax layer at the end
            # Py-Torch's implementation of Cross Entropy Loss usings LogSoftmax
            # with negative log likelihood loss
            #Softmax(dim = -1)
        )
        
        return linear_layers

    def _define_loss_function(self):
        """Defines the loss function of the model"""
        return CrossEntropyLoss()

    def _define_optimizer(self):
        """Defines the optimizer for the model."""
        return SGD(self.parameters(), lr = 0.01, momentum = 0.9, weight_decay= 5e-4)


class VGG11_Decomposed(VGG11): 
    """A subclass of the above scaled down VGG-11 Class, but returns decomposed CNN layers."""
    def __init__(self, loaders, num_classes, decomposition = Decomposition.CP): 
        self._decomposition = decomposition
        super(VGG11_Decomposed, self).__init__(loaders, num_classes)


    def _define_cnn_layers(self):
        org_cnn_layers = super(VGG11_Decomposed, self)._define_cnn_layers()
        return decompose_cnn_layers(org_cnn_layers, self._decomposition)
       