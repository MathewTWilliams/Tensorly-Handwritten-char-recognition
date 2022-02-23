# Author: Matt Williams
# Version: 2/21/2022

from matplotlib import pyplot as plt
from load_datasets import *


def visualize_images(images, label): 
    for i in range(10):
        plt.subplot(330 + 1 + i, title = label)
        plt.imshow(images[i], cmap = plt.get_cmap('gray'))

    plt.show()



if __name__ == "__main__": 
    dataset = load_testing_dataset()

    for label, images in dataset.items(): 
        visualize_images(images, label)
        