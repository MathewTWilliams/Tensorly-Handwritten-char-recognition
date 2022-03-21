# Author: Matt Williams
# Version: 3/16/2022

from constants import *
import pandas as pd



def _get_mappings(): 
    '''Get our label mappings from the provided .txt file from the original data set'''
    mappings = {}

    with open(EMNIST_MAPPING_PATH, mode = "r", encoding = "utf-8") as file: 
        lines = file.readlines()
        for line in lines: 
            mapping = line.strip().split()
            mappings[int(mapping[0])] = chr(int(mapping[1]))

    return mappings

def _edit_labels(data_set): 

    mappings = _get_mappings()
    
    for o_label, n_label in mappings.items(): 
        data_set[LABEL_COL] = data_set[LABEL_COL].replace(o_label, str(n_label))

def _split_data_set(data_set): 
    '''Split the given data set based on if the label is a number or a letter'''
    dataset_groups = data_set.groupby(data_set.columns[0])

    number_dfs = []
    letter_dfs = []

    for label in dataset_groups.groups.keys():
        if label.isalpha():
            letter_dfs.append(dataset_groups.get_group(label))
        else: 
            number_dfs.append(dataset_groups.get_group(label))
    

    return pd.concat(number_dfs), pd.concat(letter_dfs)

def _train_valid_split(train_set, n): 

    train_set_groups = train_set.groupby(train_set.columns[0])

    train_dfs = []
    valid_dfs = []

    for label in train_set_groups.groups.keys(): 
        group_df = train_set_groups.get_group(label)
        valid_df = group_df.sample(n=n, replace = False)
        valid_dfs.append(valid_df)
        group_df = group_df.drop(index = valid_df.index)
        train_dfs.append(group_df)

    return pd.concat(train_dfs), pd.concat(valid_dfs)

if __name__ == "__main__": 
    train_set = pd.read_csv(TRAINING_DATA_PATH_CSV, header = None)
    test_set = pd.read_csv(TESTING_DATA_PATH_CSV, header = None)

    # edit labels to actual characters instead of ascii values
    _edit_labels(train_set)
    _edit_labels(test_set)
    
    # calculate number of elements needed for validation set
    test_groupby = test_set.groupby(train_set.columns[0])
    valid_count = int(test_groupby.size().mean())

    test_numbers, test_letters = _split_data_set(test_set)
    test_numbers.to_csv(TESTING_DATA_PATH_NUMBERS, header=False, index=False)
    test_letters.to_csv(TESTING_DATA_PATH_LETTERS, header=False, index=False)

    train_numbers, train_letters = _split_data_set(train_set)
    
    train_numbers, valid_numbers = _train_valid_split(train_numbers, valid_count)
    train_numbers.to_csv(TRAINING_DATA_PATH_NUMBERS, header=False, index = False)
    valid_numbers.to_csv(VALIDATE_DATA_PATH_NUMBERS, header=False, index = False)

    train_letters, valid_letters = _train_valid_split(train_letters, valid_count)
    train_letters.to_csv(TRAINING_DATA_PATH_LETTERS, header=False, index = False)
    valid_letters.to_csv(VALIDATE_DATA_PATH_LETTERS, header=False, index = False)