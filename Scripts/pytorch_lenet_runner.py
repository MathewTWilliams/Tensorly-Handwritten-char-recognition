# Author: Matt Williams
# Version: 3/22/2022

from save_load_dataset import *
import torch
import matplotlib.pyplot as plt
from constants import N_LET_CLASSES, N_NUM_CLASSES
from LeNet_PyTorch import LeNet_5
from pytorch_utils import *
from torchsummary import summary


def _run_numbers():
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    alex_numbers = LeNet_5(num_data_loaders, N_NUM_CLASSES)

    if torch.cuda.is_available(): 
        alex_numbers = alex_numbers.cuda()
        alex_numbers.to_cuda()

    summary(alex_numbers, (1, 28, 28), batch_size=32)

    run_model(alex_numbers,valid_set_func=load_validation_number_dataset, name="LeNet-5 Pytorch", dataset_name="Numbers")


def _run_letters():
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    alex_letters = LeNet_5(let_data_loaders, N_LET_CLASSES)

    if torch.cuda.is_available():
        alex_letters = alex_letters.cuda()
        alex_letters.to_cuda()

    summary(alex_letters, (1, 28, 28), batch_size=32)
    run_model(alex_letters,valid_set_func=load_validation_letter_dataset, name="LeNet-5 Pytorch", dataset_name="Letters")


def _run_balanced():
    bal_data_loaders = make_data_loaders(load_training_balanced_dataset, 
                                        load_validation_balanced_dataset)

    alex_balanced = LeNet_5(bal_data_loaders, N_BAL_CLASSES)

    if torch.cuda.is_available():
        alex_balanced = alex_balanced.cuda()
        alex_balanced.to_cuda()

    summary(alex_balanced, (1, 28, 28), batch_size=32)
    run_model(alex_balanced,valid_set_func=load_validation_balanced_dataset, name="LeNet-5 Pytorch", dataset_name="Balanced")


if __name__ == "__main__": 
 
    #_run_numbers()
    #_run_letters()
    _run_balanced()

    
    
