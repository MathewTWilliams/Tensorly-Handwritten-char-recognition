# Author: Matt Williams
# Version: 3/20/2022
# reference: https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# this file is for learning purposes

from pickletools import optimize
import torch
from timeit import Timer
from save_load_dataset import *
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d
from torch.nn import MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_x, train_y, test_x, test_y):
    train_losses = []
    test_losses = []

    model = CNN()

    if torch.cuda.is_available(): 
        model = model.cuda()
        criterion = criterion.cuda()
    
    model.train()

    tr_loss = 0

    # get training training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # get the test set
    x_test, y_test = Variable(test_x), Variable(test_y)

    #convert data to GPU format
    if torch.cuda.is_available(): 
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    # zero out the gradients
    optimizer.zero_grad()

    #prediction on training and test set
    output_train = model(x_train)
    output_test = model(x_test)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_test = criterion(output_test, y_test)
    train_losses.append(loss_train)
    test_losses.append(loss_test)

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()



class CNN(Module): 
    def __init__(self):
        super(CNN, self).__init__()

        #define Model
        self.cnn_layers = Sequential(
            #Defining a 2d convolution layers
            Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride = 1, padding = 1),
            BatchNorm2d(num_features=4),
            ReLU(inplace = True), 
            MaxPool2d(kernel_size=2, stride = 2), 
            # Defining another 2D convolution layer
            Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=4),
            ReLU(inplace = True), 
            MaxPool2d(kernel_size=2, stride = 2)
        )

        self.linear_layers = Sequential(
            Linear(in_features= 4*7*7, out_features=10)
        )


        self.optimizer = Adam(self.parameters(), lr = 0.07)
        self.criterion = CrossEntropyLoss()




    def forward(self, x): 
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



if __name__ == "__main__": 
    
    train_x, train_y = load_testing_number_dataset(True, num_color_channels=1, torch=True)
    valid_x, valid_y = load_validation_number_dataset(True, num_color_channels=1, torch=True)

    
    n_epochs = 25




    

    

    




