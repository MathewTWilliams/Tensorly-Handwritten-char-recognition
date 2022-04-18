# Author: Matt Williams
# Version: 3/28/2022
# references: 
# - https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# - https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
# - https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/
# - https://github.com/JeanKossaifi/tensorly-notebooks/blob/master/05_pytorch_backend/cnn_acceleration_tensorly_and_pytorch.ipynb
# - https://github.com/jacobgil/pytorch-tensor-decompositions

import torch
from torch.nn import Module, Sequential
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from constants import *
from datetime import datetime
from sklearn.metrics import classification_report
import numpy as np
from save_results import save_cnn_results
from pytorch_decompositions import * 

class Py_Torch_Base(Module):

    def __init__(self, loaders, num_classes): 
        super(Py_Torch_Base, self).__init__()
        self._loaders = loaders
        self._num_classes = num_classes
        self._cnn_layers = self._define_cnn_layers()
        self._linear_layers = self._define_linear_layers()
        self._optimizer = self._define_optimizer()
        self._loss_function = self._define_loss_function()


    def forward(self, x):
        x = x.to(dtype=torch.float)
        x = self._cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self._linear_layers(x)
        return x
        
    def _define_cnn_layers(self):
        '''Needs to be defined by the super class'''
        pass

    def _define_linear_layers(self):
        '''needs to be defined by the super class'''
        pass

    def _define_optimizer(self):
        '''Needs to be defined by the super class'''
        pass

    def _define_loss_function(self):
        '''Needs to be defined by the super class'''
        pass

    def to_cuda(self): 
        self._loss_function = self._loss_function.cuda() 

    def run_epochs(self, n_epochs, validate = True): 
        train_time = 0
        train_losses = []
        valid_losses = []

        for _ in range(n_epochs): 
            start = datetime.now()
            train_loss = self._train(True)
            end = datetime.now()
            train_losses.append(train_loss)
            train_time += (end - start).total_seconds()

        if validate:
            for _ in range(n_epochs):
                start = datetime.now()
                valid_loss = self._train(False)
                end = datetime.now() 
                valid_losses.append(valid_loss)
                train_time += (end - start).total_seconds()

        return train_losses, valid_losses, train_time



    def _train(self, on_train = True):
        data_set_name = "train" if on_train else "valid"
        running_total = 0
        for i, (images, labels) in enumerate(self._loaders[data_set_name]): 
            b_x = Variable(images)
            b_y = Variable(labels)

            if torch.cuda.is_available():
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            self._optimizer.zero_grad()

            output = self(b_x)

            loss_train = self._loss_function(output, b_y)
            running_total += loss_train.item()

            if on_train: 
                loss_train.backward()
                self._optimizer.step()
        
        return running_total / len(self._loaders[data_set_name].dataset)


class EMNIST_Dataset(Dataset): 
    def __init__(self, data, labels): 
        self._data = data
        self._labels = labels
    
    def __len__(self): 
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]



def make_data_loaders(train_set_func, valid_set_func, normalize = True, num_color_channels = 1):

    train_x, train_y = train_set_func(normalize, num_color_channels, torch = True)
    valid_x, valid_y = valid_set_func(normalize, num_color_channels, torch = True)

    train_dataset = EMNIST_Dataset(train_x, train_y)
    valid_dataset = EMNIST_Dataset(valid_x, valid_y)

    loaders = {
        "train": DataLoader(
                            train_dataset, 
                            batch_size=BATCH_SIZE, 
                            num_workers=1, 
                            shuffle = True,
        ),

        "valid": DataLoader(
                            valid_dataset, 
                            batch_size=BATCH_SIZE, 
                            num_workers=1, 
                            shuffle = True,
        ),
    }

    return loaders

def run_predictions(model, test_x, test_y): 

    with torch.no_grad():
        start = datetime.now() 
        output = model(test_x.cuda())
        end = datetime.now()
        print("time:", (end-start).total_seconds())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis = -1)

    class_report = classification_report(test_y, predictions, output_dict=True)
    
    return class_report



def run_model(model, valid_set_func, name, dataset_name, normalize = True, num_color_channels = 1):
    train_losses, valid_losses, train_time = model.run_epochs(N_EPOCHS, VALIDATE)


    num_valid_x, num_valid_y = valid_set_func(normalize, num_color_channels, torch=True)

    num_valid_x = torch.from_numpy(num_valid_x)
    num_valid_y = torch.from_numpy(num_valid_y)


    if torch.cuda.is_available(): 
        num_valid_x = num_valid_x.cuda()
      

    class_report = run_predictions(model, num_valid_x, num_valid_y)

    model_details = model.__repr__().strip().split("\n")
    
    results_dict = {
        "Name": name, 
        "Dataset": dataset_name,
        "Model": model_details,
        "Train Loss per Epoch" : train_losses,
        "Classification Report": class_report, 
        "Training Time": train_time,
    }
    
    if VALIDATE: 
        results_dict['Valid Loss per Epoch'] = valid_losses

    save_cnn_results(results_dict, PYT_RESULTS_FOLDER)

def initialize_weights_bias(layer): 
    
    if layer.weight is not None: 
        torch.nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None: 
        torch.nn.init.zeros_(layer.bias)

def decompose_cnn_layers(cnn_layers, decomposition = Decomposition.CP): 
    
    decomposed_cnn_layers = Sequential()
    found_first_cnn = False
    for i, module in enumerate(cnn_layers.modules()):
        #Skip first module in the list as it gives overview of sub-modules 
        if i == 0: 
            continue
        
        #Skip first Convolution layer as it only has 1 input channel
        if type(module) is torch.nn.Conv2d and not found_first_cnn:
            decomposed_cnn_layers.append(module)
            found_first_cnn = True
            continue 
        
        elif type(module) is not torch.nn.Conv2d: 
            decomposed_cnn_layers.append(module)
            continue

        if decomposition == Decomposition.CP: 
            rank = max(module.weight.data.numpy().shape) // 3
            decomposed_layers = cp_decomposition_cnn_layer(module, rank = rank)
            for layer in decomposed_layers: 
                decomposed_cnn_layers.append(layer)

        elif decomposition == Decomposition.Tucker:
            ranks = estimate_tucker_ranks(module)
            #ranks =  [module.weight.size(0)//2, module.weight.size(1)//2]
            decomposed_layers = tucker_decomposition_cnn_layer(module, ranks = ranks)
            for layer in decomposed_layers: 
                decomposed_cnn_layers.append(layer)

    return decomposed_cnn_layers