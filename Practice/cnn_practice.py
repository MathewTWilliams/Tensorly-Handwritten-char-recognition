#Reference: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizer_v1 import SGD

def show_dataset(): 
    (trainX, trainY), (testX, testY) = mnist.load_data()
    print('Train: X=%s, y = %s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y = %s' % (testX.shape, testY.shape))
    
    for i in range(9): 
        plt.subplot(330 + 1 + i)
        plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

    plt.show()
    
def load_dataset(): 
    (trainX, trainY), (testX, testY) = mnist.load_data()
    #print(trainX.shape)
    trainX = trainX.reshape((trainX.shape[0],28, 28, 1))
    trainY = trainY.reshape((trainY.shape[0],28, 28, 1))
    #print(trainX.shape)

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    #print(trainY)
    #print(trainY[0])
    #print(testY)
    #print(testY[0])
    return trainX, trainY, testX, testY


def prep_pixels(train, test): 
    train_norm = train.astype('float32')
    test_norm = test.astype('float32') 

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

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
    # 
    model.add(Conv2D(32, (3,3), activation = "relu", kernel_initializer='he_uniform', input_shape = (28, 28, 1)))

    # MaxPooling 2D: 
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation = "softmax"))

    #compile model
    opt = SGD(learning_rate=0.01, momentum = 0.9)
    model.compile(optimizer=opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model
if __name__ == "__main__": 
    #show_dataset()
    load_dataset()