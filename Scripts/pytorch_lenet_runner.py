# Author: Matt Williams
# Version: 3/22/2022

from save_load_dataset import *
import torch
import matplotlib.pyplot as plt
from constants import N_LET_CLASSES, N_NUM_CLASSES, Decomposition
from LeNet_PyTorch import LeNet_5, LeNet_5_Decomposed
from pytorch_utils import *
from torchsummary import summary


def _run_numbers():
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    lenet_numbers = LeNet_5(num_data_loaders, N_NUM_CLASSES, Decomposition.CP)

    if torch.cuda.is_available(): 
        lenet_numbers = lenet_numbers.cuda()
        lenet_numbers.to_cuda()

    summary(lenet_numbers, (1, 28, 28), batch_size=32)

    run_model(lenet_numbers,valid_set_func=load_validation_number_dataset, name="LeNet-5 Pytorch", dataset_name="Numbers")
    del lenet_numbers

def _run_letters():
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    lenet_letters = LeNet_5(let_data_loaders, N_LET_CLASSES)

    if torch.cuda.is_available():
        lenet_letters = lenet_letters.cuda()
        lenet_letters.to_cuda()

    summary(lenet_letters, (1, 28, 28), batch_size=32)
    run_model(lenet_letters,valid_set_func=load_validation_letter_dataset, name="LeNet-5 Pytorch", dataset_name="Letters")
    del lenet_letters

def _run_balanced():
    bal_data_loaders = make_data_loaders(load_training_balanced_dataset, 
                                        load_validation_balanced_dataset)

    lenet_balanced = LeNet_5(bal_data_loaders, N_BAL_CLASSES)

    if torch.cuda.is_available():
        lenet_balanced = lenet_balanced.cuda()
        lenet_balanced.to_cuda()

    summary(lenet_balanced, (1, 28, 28), batch_size=32)
    run_model(lenet_balanced,valid_set_func=load_validation_balanced_dataset, name="LeNet-5 Pytorch", dataset_name="Balanced")
    del lenet_balanced

def run_lenet_pytorch(): 
    _run_numbers()
    _run_letters()
    _run_balanced()



if __name__ == "__main__": 
 
    _run_numbers()
    #_run_letters()
    #_run_balanced()
    

    
    
