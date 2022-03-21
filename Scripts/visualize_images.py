# Author: Matt Williams
# Version: 3/12/2022

from matplotlib import pyplot as plt
from save_load_dataset import load_validation_number_dataset



def visualize_images(images, label): 
    for i in range(9):
        plt.subplot(330 + 1 + i, label = label)
        plt.imshow(images[i+3000], cmap = plt.get_cmap('gray'), )

    plt.show()



if __name__ == "__main__": 
    images, labels = load_validation_number_dataset(num_color_channels=1)
    visualize_images(images, "test")
        