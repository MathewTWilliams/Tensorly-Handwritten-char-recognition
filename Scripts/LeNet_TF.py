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
from tf_utils import run_model


# Includes pylance can't confirm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD


def _define_model(num_classes): 
    model = Sequential()
    #CNN specific layers
    model.add(Conv2D(filters = 6, kernel_size = (5,5), activation = "tanh", kernel_initializer ='glorot_uniform', input_shape = (28,28,1), padding = "same", strides = 1))
    model.add(AveragePooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = "tanh", kernel_initializer ='glorot_uniform', input_shape = (14,14,6), padding = "valid", strides = 1))
    model.add(AveragePooling2D(pool_size = (2,2), strides = 2 ))
    model.add(Conv2D(filters = 120, kernel_size = (5,5), activation = "tanh", kernel_initializer ='glorot_uniform', input_shape = (7,7,16), padding = "valid", strides = 1))
    model.add(Flatten())
    #fully connected layer
    model.add(Dense(units = 84, activation = "tanh", kernel_initializer = "glorot_uniform"))
    model.add(Dense(units = num_classes, activation = "softmax", kernel_initializer = "glorot_uniform"))

    #optimizer
    opt = SGD(learning_rate = 0.01, momentum = 0.9)

    model.compile(optimizer = opt, loss = "categorical_crossentropy")

    return model

def _run_letters():
    model = _define_model(N_LET_CLASSES)
    run_model(model, load_training_letter_dataset, load_validation_letter_dataset, 
            "Lenet-5", "Letters")

    
    

def _run_numbers():
    model = _define_model(N_NUM_CLASSES)
    run_model(model, load_training_number_dataset, load_validation_number_dataset, 
            "Lenet-5", "Numbers")
    

if __name__ == "__main__":

    _run_letters()
    #_run_numbers()
