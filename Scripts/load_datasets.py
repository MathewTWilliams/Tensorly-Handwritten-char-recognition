# Author: Matt Williams
# Version: 2/21/2022

import pandas as pd
from constants import *




def _load_dataset(path):
    '''Returns a dictionary where each key is the label and each value is a list of 
    2-dimensional lists. Each 2 dimensional list represents the pixel values of an image.
    
    Example: label_dict[label][i][r][c] will give use the cth pixel in the rth row of the ith image associated 
    with the label given. '''
    dataset = pd.read_csv(path, header=None)

    label_dict = {}
    for row in dataset.values: 
        label = row[0]

        if label not in label_dict.keys(): 
            label_dict[label] = []

        pixel_values = [[0 for _ in range(PIXEL_SIDE_LENGTH)] for _ in range(PIXEL_SIDE_LENGTH)]

        for i, val in enumerate(row[1:]): 
            # pixel values stored column after column instead of row after row
            r = i % PIXEL_SIDE_LENGTH
            c = i // PIXEL_SIDE_LENGTH
            pixel_values[r][c] = val

        label_dict[label].append(pixel_values)

    return label_dict
    
def load_training_dataset(): 
    return _load_dataset(TRAINING_DATA_PATH)

def load_testing_dataset(): 
    return _load_dataset(TESTING_DATA_PATH)

# just for testing purposes
if __name__ == "__main__":
    print(load_testing_dataset()[0])
