# Author: Matt Williams
# Version: 3/22/2022

from save_load_dataset import *
import torch
import matplotlib.pyplot as plt
from constants import N_LET_CLASSES, N_NUM_CLASSES
from AlexNet_PyTorch import AlexNet
from pytorch_utils import *


def _run_numbers():
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    alex_numbers = AlexNet(num_data_loaders, N_NUM_CLASSES)

    if torch.cuda.is_available(): 
        alex_numbers = alex_numbers.cuda()
        alex_numbers.to_cuda()

    run_model(alex_numbers,valid_set_func=load_validation_number_dataset, name="AlexNet Pytorch", dataset_name="Numbers")


def _run_letters():
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    alex_letters = AlexNet(let_data_loaders, N_LET_CLASSES)

    if torch.cuda.is_available():
        alex_letters = alex_letters.cuda()
        alex_letters.to_cuda()

    run_model(alex_letters,valid_set_func=load_validation_letter_dataset, name="AlexNet Pytorch", dataset_name="Letters")

if __name__ == "__main__": 

    #_run_letters()
    _run_numbers()

    
    
