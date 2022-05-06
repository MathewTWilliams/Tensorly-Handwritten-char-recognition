# Author: Matt Williams
# Version: 3/27/2022

from matplotlib import pyplot as plt
from save_load_dataset import *
import math


def visualize_images(images, labels, num_classes, class_interval):
    '''Shows the first images of each class in a dataset'''
    sqrt = math.sqrt(num_classes)
    n_cols = math.ceil(sqrt)
    n_rows = math.floor(sqrt)


    for i in range(num_classes):
        index = i*class_interval
        plt.subplot(n_rows, n_cols, 1 + i)
        plt.title(labels[index])
        plt.imshow(images[index], cmap = plt.get_cmap('gray'),)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    '''Displays the first images of both the letters and numbers class using the above method'''
    images, labels = load_validation_number_dataset(num_color_channels=1)
    num_classes = len(set(labels))
    class_interval = int(len(labels) / num_classes)
    visualize_images(images, labels, num_classes, class_interval)

    images, labels = load_validation_letter_dataset(num_color_channels=1)
    num_classes = len(set(labels))
    class_interval = int(len(labels) / num_classes)
    visualize_images(images, labels, num_classes, class_interval)
    


        