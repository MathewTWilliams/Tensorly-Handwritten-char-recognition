# Author: Matt Williams
# Version: 3/22/2022

from save_load_dataset import *
import torch
import matplotlib.pyplot as plt
from constants import N_LET_CLASSES, N_NUM_CLASSES
from VGG11_PyTorch import VGG11, VGG11_Decomposed
from pytorch_utils import *
from torchsummary import summary


def _run_numbers():
    '''Method to run the VGG-11 model on the numbers dataset.'''
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    vgg_11_numbers = VGG11(num_data_loaders, N_NUM_CLASSES)

    if torch.cuda.is_available(): 
        vgg_11_numbers = vgg_11_numbers.cuda()
        vgg_11_numbers.to_cuda()

    summary(vgg_11_numbers, (1, 28, 28), batch_size=32)

    run_model(vgg_11_numbers,valid_set_func=load_testing_number_dataset, name="VGG-11 Pytorch", dataset_name="Numbers")
    del vgg_11_numbers

def _run_letters():
    '''Method to run the VGG-11 model on the letters dataset.'''
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    vgg_11_letters = VGG11(let_data_loaders, N_LET_CLASSES)

    if torch.cuda.is_available():
        vgg_11_letters = vgg_11_letters.cuda()
        vgg_11_letters.to_cuda()

    summary(vgg_11_letters, (1, 28, 28), batch_size=32)
    run_model(vgg_11_letters,valid_set_func=load_testing_letter_dataset, name="VGG-11 Pytorch", dataset_name="Letters")
    del vgg_11_letters

def _run_balanced():
    '''Method to run the VGG-11 model on the entire dataset.'''
    bal_data_loaders = make_data_loaders(load_training_balanced_dataset, 
                                        load_validation_balanced_dataset)

    vgg_11_balanced = VGG11(bal_data_loaders, N_BAL_CLASSES)

    if torch.cuda.is_available():
        vgg_11_balanced = vgg_11_balanced.cuda()
        vgg_11_balanced.to_cuda()

    summary(vgg_11_balanced, (1, 28, 28), batch_size=32)
    run_model(vgg_11_balanced,valid_set_func=load_testing_balanced_dataset, name="VGG-11 Pytorch", dataset_name="Balanced")
    del vgg_11_balanced


def run_vgg_11_pytorch(): 
    '''Method used to run all variations of the VGG-11 model in Py-Torch'''
    _run_numbers()
    _run_letters()
    _run_balanced()


def _run_numbers_decomposed(decomposition):
    '''Given the type of decomposition, run the Decomposed VGG-11 model on the numbers dataset.'''
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    vgg_11_numbers = VGG11_Decomposed(num_data_loaders, N_NUM_CLASSES, decomposition)

    if torch.cuda.is_available(): 
        vgg_11_numbers = vgg_11_numbers.cuda()
        vgg_11_numbers.to_cuda()

    summary(vgg_11_numbers, (1, 28, 28), batch_size=32)
    name = get_decomp_name(decomposition) + "-VGG-11 Pytorch"
    run_model(vgg_11_numbers,valid_set_func=load_testing_number_dataset, name=name, dataset_name="Numbers")
    del vgg_11_numbers

def _run_letters_decomposed(decomposition):
    '''Given the type of decomposition, run the Decomposed VGG-11 model on the letters dataset.'''
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    vgg_11_letters = VGG11_Decomposed(let_data_loaders, N_LET_CLASSES, decomposition)

    if torch.cuda.is_available():
        vgg_11_letters = vgg_11_letters.cuda()
        vgg_11_letters.to_cuda()

    summary(vgg_11_letters, (1, 28, 28), batch_size=32)
    name = get_decomp_name(decomposition) + "-VGG-11 Pytorch"
    run_model(vgg_11_letters,valid_set_func=load_testing_letter_dataset, name=name, dataset_name="Letters")
    del vgg_11_letters

def _run_balanced_decomposed(decomposition):
    '''Given the type of decomposition, run the Decomposed VGG-11 model on the entire dataset.'''
    bal_data_loaders = make_data_loaders(load_training_balanced_dataset, 
                                        load_validation_balanced_dataset)

    vgg_11_balanced = VGG11_Decomposed(bal_data_loaders, N_BAL_CLASSES, decomposition)

    if torch.cuda.is_available():
        vgg_11_balanced = vgg_11_balanced.cuda()
        vgg_11_balanced.to_cuda()

    summary(vgg_11_balanced, (1, 28, 28), batch_size=32)
    name = get_decomp_name(decomposition) + "-VGG-11 Pytorch"
    run_model(vgg_11_balanced,valid_set_func=load_testing_balanced_dataset, name=name, dataset_name="Balanced")
    del vgg_11_balanced

def run_decomp_vgg11_pytorch(decomposition = Decomposition.CP): 
    '''Given the type of decomposition,to run all variations of the Decomposed VGG-11 model in Py-Torch'''
    _run_numbers_decomposed(decomposition)
    _run_letters_decomposed(decomposition)
    _run_balanced_decomposed(decomposition)

if __name__ == "__main__": 
    
    run_vgg_11_pytorch()
    #run_decomp_vgg11_pytorch(Decomposition.CP)
    #run_decomp_vgg11_pytorch(Decomposition.Tucker)

    
    
