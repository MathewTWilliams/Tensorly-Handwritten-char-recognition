#Author: Matt Williams
#Version: 3/24/2022
# Architecture Reference: 
# https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/


import tensorflow as tf
from save_load_dataset import *
from constants import *
from tf_utils import run_model


# Includes pylance can't confirm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD


def _define_model(num_classes): 
    model = Sequential()
    #CNN specific layers
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = "relu", kernel_initializer ='glorot_uniform', input_shape = (28,28,1), padding = "valid", strides = 1))
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = "relu", kernel_initializer ='glorot_uniform', input_shape = (16,16,32), padding = "same", strides = 1))
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), kernel_initializer ='glorot_uniform', input_shape = (6,6,64), padding = "same", strides = 1))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), kernel_initializer ='glorot_uniform', input_shape = (6,6,128), padding = "same", strides = 1))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_uniform', input_shape = (6,6,128), padding = "same", strides = 1))
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Dropout(rate = 0.3))
    model.add(Flatten())
    #fully connected layer
    model.add(Dense(units = 256, activation = "relu", kernel_initializer = "glorot_uniform"))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(units = 64, activation = "relu", kernel_initializer = "glorot_uniform"))
    model.add(Dense(units = num_classes, activation = "softmax", kernel_initializer = "glorot_uniform"))

    #optimizer
    opt = SGD(learning_rate = 0.01, momentum = 0.9)

    model.compile(optimizer = opt, loss = "categorical_crossentropy")

    return model

def _run_letters():
    model = _define_model(N_LET_CLASSES)
    run_model(model, load_training_letter_dataset, load_validation_letter_dataset, 
            "AlexNet", "Letters")


def _run_numbers():
    model = _define_model(N_NUM_CLASSES)
    run_model(model, load_training_number_dataset, load_validation_number_dataset, 
            "AlexNet", "Numbers")

def _run_balanced():
    model = _define_model(N_BAL_CLASSES)
    run_model(model, load_training_balanced_dataset, load_validation_balanced_dataset, 
            "AlexNet", "Balanced")



if __name__ == "__main__": 
    #_run_letters()
    #_run_numbers()
    _run_balanced()

