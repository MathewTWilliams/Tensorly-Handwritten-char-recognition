# Author: Matt Williams
# Version: 4/2/2022
# Architecture Reference: 
# https://arxiv.org/pdf/1409.1556.pdf



import tensorflow as tf
from save_load_dataset import *
from constants import *
from tf_utils import run_model


# Includes pylance can't confirm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2


def _define_model(num_classes): 
    model = Sequential()
    
    #Convolution Layers
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (28,28,1), padding = "same", strides = 1))

    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (14,14,32), padding = "same", strides = 1))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (8,8,64), padding = "same", strides = 1))

    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (8,8,128), padding = "same", strides = 1))

    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (4,4,128), padding = "same", strides = 1))


    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (4,4,256), padding = "same", strides = 1))
    
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (2,2,256), padding = "same", strides = 1))

    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
                    kernel_regularizer = l2(5e-4), input_shape = (2,2,256), padding = "same", strides = 1))

    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))


    # Linear Layers
    model.add(Flatten())
    model.add(Dense(units = 128, activation = "relu", kernel_initializer = "glorot_normal", kernel_regularizer = l2(5e-4)))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(units = 64, activation = "relu", kernel_initializer = "glorot_normal", kernel_regularizer = l2(5e-4)))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(units = num_classes, activation = "softmax", kernel_initializer = "glorot_normal", kernel_regularizer = l2(5e-4)))

    opt = SGD(learning_rate = 0.01, momentum = 0.9)
    model.compile(optimizer = opt, loss = "categorical_crossentropy")

    return model

def _run_letters():
    model = _define_model(N_LET_CLASSES)
    run_model(model, load_training_letter_dataset, load_validation_letter_dataset, 
            "VGG-11", "Letters")
    del model

def _run_numbers():
    model = _define_model(N_NUM_CLASSES)
    run_model(model, load_training_number_dataset, load_validation_number_dataset, 
            "VGG-11", "Numbers")
    del model

def _run_balanced():
    model = _define_model(N_BAL_CLASSES)
    run_model(model, load_training_balanced_dataset, load_validation_balanced_dataset, 
            "VGG-11", "Balanced")
    del model


def run_vgg_11_tf_models(): 
    _run_numbers()
    _run_letters()
    _run_balanced()
    


if __name__ == "__main__":

    #_run_letters()
    #_run_numbers()
    _run_balanced()

