# Author: Matt Williams
# Version: 3/12/2022

#reference: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
#This file is for learning purposes

import tensorflow as tf
from save_load_dataset import load_training_number_dataset
from save_load_dataset import load_validation_number_dataset
from timeit import Timer


#Includes not reconized by pylance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical




def define_model(): 
    # sequential model appropriate for a plain stack of layers where each
    # layer has exactly one input tensor and one output tensor
    model = Sequential()

    # Conv2D: used for Spatial convolutions over images
    # - produces a convolution kernel and uses that kernel to return a tensor of outputs
    # Activation layer: applies an activation function to the input received by previous neurons 
    # - done to keep values within manageable range. 
    # ReLu: return weighted_sum <= 0 ? 0 : weighted_sum
    # he_uniform: Variance scaling initializer, 
    # - draws samples from a uniform distribution(probability distribution with constant probability) within [-limit, limit], 
    # - where limit = sqrt(6/fan_in)
    # - where fan_in is number of input units in the weight tensor
    model.add(Conv2D(32, (3,3), activation = "relu", kernel_initializer='he_uniform', input_shape = (28, 28, 1)))
    
    # Downsamples the input along its height and width by taking the max value
    # over an input window (2,2) here, goes across the image like the convolution kernel
    # strides here is 2,2 same as input window since it isn't set
    model.add(MaxPooling2D((2,2)))

    # flattens the input given to it (turns it into a 1-d array)
    model.add(Flatten())

    # Regular Dense NN Layer, 100 neurons, each takes in the outputs from the previous layer of neurons,
    # each gives an output
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))

    # Another dense layer, 10 neurons, one for each letter?
    # - softmax function: normalizes input from previous layer to produce a probability distribution
    model.add(Dense(10, activation = "softmax"))

    # Stochastic Gradient Descent: optimization function that tries to find global minima for an objective function
    # learning_rate: hyperparameter that determines how much the weights change after each iteration
    # momentum: hyperparameter that determines how much GD will accelerate in the direction of the gradient
    opt = SGD(learning_rate = 0.01, momentum = 0.9)

    # Loss Function: function that determines how well the neural network as learned the target function
    # Cross Entropy: measures difference between an estimated probability distribution and the target probability distribution
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'])

    return model

def run_model(): 

    train_x, train_y = load_training_number_dataset(normalize=True, num_color_channels=1)
    valid_x, valid_y = load_validation_number_dataset(normalize=True, num_color_channels=1)

    train_y = to_categorical(train_y)
    valid_y = to_categorical(valid_y)

    model = define_model()
    model.fit(train_x, train_y, epochs = 10, batch_size = 32, validation_data = (valid_x, valid_y), verbose = 0)
    _, acc = model.evaluate(valid_x, valid_y, verbose = 0)
    print('> %.3f' % (acc * 100.0))
    


if __name__ == "__main__": 
    t = Timer(lambda: run_model())
    print(t.timeit(number = 1))
    


    
    
