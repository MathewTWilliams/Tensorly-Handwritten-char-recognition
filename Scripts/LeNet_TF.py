#Author: Matt Williams
#Version: 3/24/2022
# Architecture Reference: 
# https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
# http://www2.cs.uh.edu/~ceick/7362/Arthur1.pdf 
# https://en.wikipedia.org/wiki/LeNet 
# https://towardsai.net/p/deep-learning/the-architecture-and-implementation-of-lenet-5


import tensorflow as tf
from save_load_dataset import *
from constants import *
from tf_utils import run_model, compile_model
from tensorflow_decompositions import decompose_cnn_layers


# Includes pylance can't confirm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD


def _define_model(num_classes): 
    model = Sequential()
    #CNN specific layers
    model.add(Conv2D(filters = 6, kernel_size = (5,5), activation = "tanh", kernel_initializer ='glorot_normal', input_shape = (28,28,1), padding = "same", strides = 1))
    model.add(AveragePooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = "tanh", kernel_initializer ='glorot_normal', input_shape = (14,14,6), padding = "valid", strides = 1))
    model.add(AveragePooling2D(pool_size = (2,2), strides = 2 ))
    model.add(Conv2D(filters = 120, kernel_size = (5,5), activation = "tanh", kernel_initializer ='glorot_normal', input_shape = (7,7,16), padding = "valid", strides = 1))
    model.add(Flatten())
    #fully connected layer
    model.add(Dense(units = 84, activation = "tanh", kernel_initializer = "glorot_normal"))
    model.add(Dense(units = num_classes, activation = "softmax", kernel_initializer = "glorot_normal"))

    #optimizer
    return model

def _run_letters():
    model = _define_model(N_LET_CLASSES)
    model = compile_model(model)
    run_model(model, load_training_letter_dataset, load_validation_letter_dataset, 
            "Lenet-5 TF", "Letters")
    del model
    
def _run_numbers():
    model = _define_model(N_NUM_CLASSES)
    model = compile_model(model)
    run_model(model, load_training_number_dataset, load_validation_number_dataset, 
            "Lenet-5 TF", "Numbers")
    del model

def _run_balanced():
    model = _define_model(N_BAL_CLASSES)
    model = compile_model(model)
    run_model(model, load_training_balanced_dataset, load_validation_balanced_dataset, 
            "Lenet-5 TF", "Balanced")
    del model
    
def run_lenet_tf_models(): 
    _run_numbers()
    _run_letters()
    _run_balanced()


def _run_letters_decomposed(decomposition):
    model = _define_model(N_LET_CLASSES)
    model = decompose_cnn_layers(model, decomposition)
    model = compile_model(model)
    name = get_decomp_name(decomposition) + "-LeNet-5 TF"
    run_model(model, load_training_letter_dataset, load_validation_letter_dataset, 
            name, "Letters")
    del model

def _run_numbers_decomposed(decomposition):
    model = _define_model(N_NUM_CLASSES)
    model = decompose_cnn_layers(model, decomposition)
    model = compile_model(model)
    name = get_decomp_name(decomposition) + "-LeNet-5 TF"
    run_model(model, load_training_number_dataset, load_validation_number_dataset, 
            name, "Numbers")
    del model

def _run_balanced_decomposed(decomposition):
    model = _define_model(N_BAL_CLASSES)
    model = decompose_cnn_layers(model, decomposition)
    model = compile_model(model)
    name = get_decomp_name(decomposition) + "-LeNet-5 TF"
    run_model(model, load_training_balanced_dataset, load_validation_balanced_dataset, 
            name, "Balanced")
    del model

def run_lenet_tf_decomposed(decomposition = Decomposition.CP): 
    _run_numbers_decomposed(decomposition)
    _run_letters_decomposed(decomposition)
    _run_balanced_decomposed(decomposition)


if __name__ == "__main__":

    #run_lenet_tf_models()
    #run_lenet_tf_decomposed(Decomposition.CP)
    #run_lenet_tf_decomposed(Decomposition.Tucker)

    _run_numbers_decomposed(Decomposition.CP)
    #_run_numbers()