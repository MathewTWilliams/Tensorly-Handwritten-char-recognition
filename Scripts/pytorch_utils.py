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
from datetime import date, datetime
from sklearn.metrics import classification_report
import numpy as np
from save_results import save_cnn_results 
import torch.nn.functional as F

class Py_Torch_Base(Module):
    """Base class for all Py-Torch models"""
    def __init__(self, loaders, num_classes): 
        super(Py_Torch_Base, self).__init__()
        self._loaders = loaders
        self._num_classes = num_classes
        self._cnn_layers = self._define_cnn_layers()
        self._linear_layers = self._define_linear_layers()
        self._optimizer = self._define_optimizer()
        self._loss_function = self._define_loss_function()
        self._back_prop_time = 0


    def forward(self, x):
        """Overriden Forward Pass method"""
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
        """Method to use if cuda is detected""" 
        self._loss_function = self._loss_function.cuda() 

    def run_epochs(self, n_epochs, validate = True): 
        """ Run the model through the given number of epochs or until validation loss increases"""
        train_time = 0
        train_losses = []
        valid_losses = []


        for i in range(n_epochs): 
            start = datetime.now()
            train_loss = self._train(True)
            train_losses.append(train_loss)
            end = datetime.now()
            train_time += (end - start).total_seconds()
            
            if(validate): 
                start = datetime.now()
                valid_loss = self._train(False)
                end = datetime.now()
                train_time += (end - start).total_seconds()
                prev_val_loss = valid_losses[-1] if len(valid_losses) > 0 else 1
                valid_losses.append(valid_loss)
                if valid_loss > prev_val_loss:
                    break

        back_prop_time = self._back_prop_time
        self._back_prop_time = 0
        return train_losses, valid_losses, train_time, back_prop_time



    def _train(self, on_train = True):
        '''Method runs the model through a single epoch'''
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
                start = datetime.now()
                loss_train.backward()
                self._optimizer.step()
                end = datetime.now()
                self._back_prop_time += (end - start).total_seconds()
        
        return running_total / len(self._loaders[data_set_name].dataset)


class EMNIST_Dataset(Dataset):
    '''Small class to hold our data set for use on Py-Torch models.'''
    def __init__(self, data, labels): 
        self._data = data
        self._labels = labels
    
    def __len__(self): 
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]



def make_data_loaders(train_set_func, valid_set_func, normalize = True, num_color_channels = 1):
    '''Make our Py-Torch Data loaders'''
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
    '''Given a model and a test set, make a classification report on the results'''
    with torch.no_grad():
        start = datetime.now() 
        output = model(test_x.cuda())
        end = datetime.now()

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis = -1)

    class_report = classification_report(test_y, predictions, output_dict=True)
    class_report['# Predictions'] = len(test_y)
    class_report["Prediction Time"] = (end-start).total_seconds()
    
    return class_report



def run_model(model, valid_set_func, name, dataset_name, normalize = True, num_color_channels = 1):
    '''Method used to train the given model on the given dataset information, fit the model and test it. Saves results'''
    train_losses, valid_losses, train_time, back_prop_time = model.run_epochs(N_EPOCHS, VALIDATE)


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
        "Back Propogation Time": back_prop_time, 
        "Epochs": len(train_losses)
    }
    
    if VALIDATE: 
        results_dict['Valid Loss per Epoch'] = valid_losses

    save_cnn_results(results_dict, PYT_RESULTS_FOLDER)

def initialize_weights_bias(layer): 
    '''Given a layer, intializer the weights and biases'''
    if layer.weight is not None: 
        torch.nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None: 
        torch.nn.init.zeros_(layer.bias)

