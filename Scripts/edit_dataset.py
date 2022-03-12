# Author: Matt Williams
# Version: 3/12/2022

from constants import *
import pandas as pd
import numpy as np
from save_load_dataset import save_json
import random

def _dataset_from_csv_to_json(data_set): 

    label_dict = {}
    for row in data_set.values: 
        label = row[0]

        if label not in label_dict.keys(): 
            label_dict[label] = []

        pixel_values = np.zeros((PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH), dtype=np.int8)

        for i, val in enumerate(row[1:]): 
            # pixel values stored column after column instead of row after row
            r = i % PIXEL_SIDE_LENGTH
            c = i // PIXEL_SIDE_LENGTH
            pixel_values[r][c] = val

        label_dict[label].append(pixel_values.tolist())

    return label_dict


def _get_mappings(): 

    mappings = {}

    with open(EMNIST_MAPPING_PATH, mode = "r", encoding = "utf-8") as file: 
        lines = file.readlines()
        for line in lines: 
            mapping = line.strip().split()
            mappings[int(mapping[0])] = chr(int(mapping[1]))

    return mappings

def _edit_labels(data_set): 

    mappings = _get_mappings()
    label_col = 0    
    
    for o_label, n_label in mappings.items(): 
        data_set[label_col] = data_set[label_col].replace(o_label, str(n_label))

def _split_data_set(data_set): 
    numbers_set = {}
    letters_set = {}

    for label, pixel_list in data_set.items(): 
        if label.isalpha():
            letters_set[label] = pixel_list
        else: 
            numbers_set[label] = pixel_list


    return numbers_set, letters_set


def _train_valid_split(train_set, num_to_split): 
    new_train_set = {label:[] for label in train_set.keys()}
    valid_set = {}

    for label in train_set.keys(): 
        valid_set[label] = random.sample(train_set[label], k= num_to_split)
        for pixel_values in train_set[label]: 
            if pixel_values not in valid_set[label]: 
                new_train_set[label].append(pixel_values)

    return new_train_set, valid_set


if __name__ == "__main__": 
    train_set = pd.read_csv(TRAINING_DATA_PATH_CSV, header = None)
    test_set = pd.read_csv(TESTING_DATA_PATH_CSV, header = None)

    # edit labels to actual characters instead of ascii values
    _edit_labels(train_set)
    _edit_labels(test_set)

    # convert to json format
    train_set = _dataset_from_csv_to_json(train_set)
    test_set = _dataset_from_csv_to_json(test_set)
    
    # calculate number of elements needed for validation set
    count = 0
    for label in test_set.keys(): 
        count += len(test_set[label])
    
    count //= len(test_set.keys())

    # split the test set and save it
    test_numbers_set, test_letters_set = _split_data_set(test_set)
    save_json(test_numbers_set, TESTING_DATA_PATH_NUMBERS)
    save_json(test_letters_set, TESTING_DATA_PATH_LETTERS)

    # split the training set
    train_numbers_set, train_letters_set = _split_data_set(train_set)

    #split training number set
    train_numbers_set, valid_numbers_set = _train_valid_split(train_numbers_set, count)
    
    #split training letter set
    train_letters_set, valid_letters_set = _train_valid_split(train_letters_set, count)

    save_json(train_numbers_set, TRAINING_DATA_PATH_NUMBERS)
    save_json(train_letters_set, TRAINING_DATA_PATH_LETTERS)

    save_json(valid_numbers_set, VALIDATE_DATA_PATH_NUMBERS)
    save_json(valid_letters_set, VALIDATE_DATA_PATH_LETTERS)

