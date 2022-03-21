# Author: Matt Williams
# Version: 3/12/2022

import numpy as np
from constants import *
import os
import pandas as pd

def _load_csv(path, normalize, num_color_channels, torch): 
    if not os.path.exists(path):
        return None

    file_path_split = path.split(".")

    #the last element in the split should be the file extention
    if not file_path_split[-1] == "csv": 
        return None

    
    csv_df = pd.read_csv(path, header = None)
    labels = csv_df[LABEL_COL].to_numpy(dtype=np.uint8)
    images = csv_df.iloc[: , (LABEL_COL + 1):].to_numpy(dtype=np.uint8)
    length = len(images)
    #used Fortan based indexing since values are store column by column instead of row by row
    images = images.reshape((length, PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH), order = "F")

    if normalize: 
        images = images / 255.0

    if num_color_channels > 0: 
        if not torch:
            images = images.reshape(length, PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH, num_color_channels)
        else: 
            images = images.reshape((length, num_color_channels, PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH))
            
    return images, labels


def save_csv(csv_df, path):
    if path.split('.')[-1] == "csv":
        if os.path.exists(path): 
            os.remove(path)
        csv_df.to_csv(path, header = False, index = False)
    

def load_training_letter_dataset(normalize = False, num_color_channels = 0, torch = False): 
    return _load_csv(TRAINING_DATA_PATH_LETTERS, normalize, num_color_channels, torch)

def load_training_number_dataset(normalize = False, num_color_channels = 0, torch = False): 
    return _load_csv(TRAINING_DATA_PATH_NUMBERS, normalize, num_color_channels, torch)

def load_testing_letter_dataset(normalize = False, num_color_channels = 0, torch = False): 
    return _load_csv(TESTING_DATA_PATH_LETTERS, normalize, num_color_channels, torch)

def load_testing_number_dataset(normalize = False, num_color_channels = 0, torch = False): 
    return _load_csv(TESTING_DATA_PATH_NUMBERS, normalize, num_color_channels, torch)

def load_validation_letter_dataset(normalize = False, num_color_channels = 0, torch = False): 
    return _load_csv(VALIDATE_DATA_PATH_LETTERS, normalize, num_color_channels, torch)

def load_validation_number_dataset(normalize = False, num_color_channels = 0, torch = False): 
    return _load_csv(VALIDATE_DATA_PATH_NUMBERS, normalize, num_color_channels, torch)

# just for testing purposes
if __name__ == "__main__":
    images, labels = load_testing_number_dataset(normalize=True, num_color_channels=1)
    
    
