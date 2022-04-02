# Author: Matt Williams
# Version: 3/22/2022

from save_load_dataset import *
import torch
import matplotlib.pyplot as plt
from constants import N_LET_CLASSES, N_NUM_CLASSES
from VGG11_PyTorch import VGG11
from pytorch_utils import *


def _run_numbers():
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    vgg_11_numbers = VGG11(num_data_loaders, N_NUM_CLASSES)

    if torch.cuda.is_available(): 
        vgg_11_numbers = vgg_11_numbers.cuda()
        vgg_11_numbers.to_cuda()

    run_model(vgg_11_numbers,valid_set_func=load_validation_number_dataset, name="VGG-11 Pytorch", dataset_name="Numbers")


def _run_letters():
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    vgg_11_letters = VGG11(let_data_loaders, N_LET_CLASSES)

    if torch.cuda.is_available():
        vgg_11_letters = vgg_11_letters.cuda()
        vgg_11_letters.to_cuda()

    run_model(vgg_11_letters,valid_set_func=load_validation_letter_dataset, name="VGG-11 Pytorch", dataset_name="Letters")

if __name__ == "__main__": 

    #_run_letters()
    _run_numbers()

    
    
