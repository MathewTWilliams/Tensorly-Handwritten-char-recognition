# Author: Matt Williams
# Version: 3/27/2022

from matplotlib import pyplot as plt
from save_load_dataset import *
import math
from utils import edit_labels


def visualize_images(images, labels, num_classes, class_interval): 
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

    images, labels = load_validation_number_dataset(num_color_channels=1)
    num_classes = len(set(labels))
    class_interval = int(len(labels) / num_classes)
    visualize_images(images, labels, num_classes, class_interval)

    images, labels = load_validation_letter_dataset(num_color_channels=1)
    labels = edit_labels(labels, False)
    num_classes = len(set(labels))
    class_interval = int(len(labels) / num_classes)
    visualize_images(images, labels, num_classes, class_interval)
    


        