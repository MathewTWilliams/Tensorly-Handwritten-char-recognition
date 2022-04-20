# Author: Matt Williams
# Version: 3/22/2022

from save_load_dataset import *
import torch
import matplotlib.pyplot as plt
from constants import N_LET_CLASSES, N_NUM_CLASSES, N_BAL_CLASSES, Decomposition, get_decomp_name
from AlexNet_PyTorch import AlexNet, AlexNet_Decomposed
from pytorch_utils import *
from torchsummary import summary


def _run_numbers():
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    alex_numbers = AlexNet(num_data_loaders, N_NUM_CLASSES)

    if torch.cuda.is_available(): 
        alex_numbers = alex_numbers.cuda()
        alex_numbers.to_cuda()

    summary(alex_numbers, input_size=(1, 28, 28), batch_size=32, device="cuda")
    run_model(alex_numbers,valid_set_func=load_validation_number_dataset, name="AlexNet Pytorch", dataset_name="Numbers")
    del alex_numbers

def _run_letters():
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    alex_letters = AlexNet(let_data_loaders, N_LET_CLASSES)

    if torch.cuda.is_available():
        alex_letters = alex_letters.cuda()
        alex_letters.to_cuda()

    summary(alex_letters, input_size=(1, 28, 28), batch_size=32, device="cuda")
    run_model(alex_letters,valid_set_func=load_validation_letter_dataset, name="AlexNet Pytorch", dataset_name="Letters")
    del alex_letters

def _run_balanced():
    bal_data_loaders = make_data_loaders(load_training_balanced_dataset, 
                                        load_validation_balanced_dataset)

    alex_balanced = AlexNet(bal_data_loaders, N_BAL_CLASSES)

    if torch.cuda.is_available():
        alex_balanced = alex_balanced.cuda()
        alex_balanced.to_cuda()

    summary(alex_balanced, input_size=(1, 28, 28), batch_size=32, device="cuda")
    run_model(alex_balanced,valid_set_func=load_validation_balanced_dataset, name="AlexNet Pytorch", dataset_name="Balanced")
    del alex_balanced


def run_alexnet_pytorch():
    _run_numbers() 
    _run_letters()
    _run_balanced()


def _run_numbers_decomposed(decomposition): 
    num_data_loaders = make_data_loaders(load_training_number_dataset, 
                                        load_validation_number_dataset)
    
    alex_numbers = AlexNet_Decomposed(num_data_loaders, N_NUM_CLASSES, decomposition)

    if torch.cuda.is_available(): 
        alex_numbers = alex_numbers.cuda()
        alex_numbers.to_cuda()

    summary(alex_numbers, input_size=(1, 28, 28), batch_size=32, device="cuda")
    name = get_decomp_name(decomposition) + "-AlexNet Pytorch"
    run_model(alex_numbers,valid_set_func=load_validation_number_dataset, name=name, dataset_name="Numbers")
    del alex_numbers

def _run_letters_decomposed(decomposition):
    let_data_loaders = make_data_loaders(load_training_letter_dataset, 
                                        load_validation_letter_dataset)

    alex_letters = AlexNet_Decomposed(let_data_loaders, N_LET_CLASSES, decomposition)

    if torch.cuda.is_available():
        alex_letters = alex_letters.cuda()
        alex_letters.to_cuda()

    summary(alex_letters, input_size=(1, 28, 28), batch_size=32, device="cuda")
    name = get_decomp_name(decomposition) + "-AlexNet Pytorch"
    run_model(alex_letters,valid_set_func=load_validation_letter_dataset, name=name, dataset_name="Letters")
    del alex_letters

def _run_balanced_decomposed(decomposition): 
    bal_data_loaders = make_data_loaders(load_training_balanced_dataset, 
                                        load_validation_balanced_dataset)

    alex_balanced = AlexNet_Decomposed(bal_data_loaders, N_BAL_CLASSES, decomposition)

    if torch.cuda.is_available():
        alex_balanced = alex_balanced.cuda()
        alex_balanced.to_cuda()

    summary(alex_balanced, input_size=(1, 28, 28), batch_size=32, device="cuda")
    name = get_decomp_name(decomposition) + "-AlexNet Pytorch"
    run_model(alex_balanced,valid_set_func=load_validation_balanced_dataset, name=name, dataset_name="Balanced")
    del alex_balanced

def run_decomp_alexnet_pytorch(decomposition = Decomposition.CP): 
    _run_numbers_decomposed(decomposition)
    _run_letters_decomposed(decomposition)
    _run_balanced_decomposed(decomposition)

if __name__ == "__main__": 
    
    #run_alexnet_pytorch()
    #run_decomp_alexnet_pytorch(Decomposition.CP)
    #run_decomp_alexnet_pytorch(Decomposition.Tucker)
    _run_numbers_decomposed(Decomposition.CP)


    
