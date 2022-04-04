# Author: Matt Williams
# Version 3/22/2022
# Architecture Reference: 
# https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
# http://www2.cs.uh.edu/~ceick/7362/Arthur1.pdf 
# https://en.wikipedia.org/wiki/LeNet 
# https://towardsai.net/p/deep-learning/the-architecture-and-implementation-of-lenet-5


import torch
from torch.nn import Linear, Conv2d, AvgPool2d, Sequential
from torch.nn import Tanh, Softmax, CrossEntropyLoss
from torch.optim import SGD
from pytorch_utils import Py_Torch_Base, initialize_weights_bias



class LeNet_5(Py_Torch_Base):

    def __init__(self, loaders, num_classes): 
        super(LeNet_5, self).__init__(loaders, num_classes)
        

    def _define_cnn_layers(self):
        #build model
        conv_1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = 2, stride = 1)
        initialize_weights_bias(conv_1)

        conv_2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding = 0, stride = 1)
        initialize_weights_bias(conv_2)

        conv_3 = Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding = 0, stride = 1)
        initialize_weights_bias(conv_3)

        cnn_layers = Sequential(
            conv_1, 
            Tanh(),
            AvgPool2d(kernel_size=2, stride = 2), 
            conv_2, 
            Tanh(), 
            AvgPool2d(kernel_size=2, stride=2), 
            conv_3, 
            Tanh(),
        )

        return cnn_layers

    def _define_linear_layers(self):
        linear_1 = Linear(120, out_features=84)
        initialize_weights_bias(linear_1)


        linear_2 = Linear(84, out_features=self._num_classes)
        initialize_weights_bias(linear_2)

        linear_layers = Sequential(
            linear_1, 
            Tanh(), 
            linear_2, 
            # No Softmax layer at the end
            # Py-Torch's implementation of Cross Entropy Loss usings LogSoftmax
            # with negative log likelihood loss
            #Softmax(dim = -1)
        )

        return linear_layers
        

    def _define_optimizer(self):
        return SGD(self.parameters(), lr = 0.01, momentum=0.9)

    def _define_loss_function(self):
        return CrossEntropyLoss()
        

