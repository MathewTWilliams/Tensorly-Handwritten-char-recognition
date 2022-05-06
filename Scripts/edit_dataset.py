# Author: Matt Williams
# Version: 3/16/2022

from constants import *
import pandas as pd
from save_load_dataset import read_mappings



def _split_data_set(data_set): 
    '''Split the given data set based on if the label is a number or a letter'''
    dataset_groups = data_set.groupby(data_set.columns[0])

    number_dfs = []
    letter_dfs = []

    for label in dataset_groups.groups.keys():
        if label < 10:
            number_dfs.append(dataset_groups.get_group(label))
        else: 
            letter_dfs.append(dataset_groups.get_group(label))

    return pd.concat(number_dfs), pd.concat(letter_dfs)

def _train_valid_split(train_set, n): 
    '''Given the training set as a data frame and a number n, splits the training set into a 
    smaller training set and a validation that that has n samples. Returns two data frames, first
    the new training set, then the validation set.'''
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
    

def _make_updated_letter_mappings(): 
    '''Makes a new class mapping file for the standalone letters dataset'''
    mappings = read_mappings(EMNIST_MAPPING_PATH)

    updated_letter_mappings = []
    for i , (label, ascii) in enumerate(mappings):
        if i < N_NUM_CLASSES: 
            continue
        new_label = int(label) - N_NUM_CLASSES
        updated_letter_mappings.append((new_label, ascii))

    with open(UPDATED_LET_MAPPINGS_PATH, mode = "w+", encoding='utf-8') as file:
        
        for new_label, ascii in updated_letter_mappings:
            file.write(str(new_label) + " " + ascii + "\n")

    return updated_letter_mappings

def _update_letter_labels(letters_df, updated_letter_mappings):

    '''Given a data frame of the letters dataset and the updated letter mappings,
        update the letter class labels with their new mappings.'''
    for updated_label, _ in updated_letter_mappings:
        old_label = updated_label + N_NUM_CLASSES
        letters_df[LABEL_COL] = letters_df[LABEL_COL].replace(old_label, updated_label)
    
    return letters_df

if __name__ == "__main__": 

    """Main method to edit the dataset into its final state"""
    # make updated letter mapping dictionary
    updated_letter_mappings = _make_updated_letter_mappings()

    # load in dataset
    train_set = pd.read_csv(TRAINING_DATA_PATH_CSV, header = None)
    test_set = pd.read_csv(TESTING_DATA_PATH_CSV, header = None)

    # calculate number of elements needed for validation set
    test_groupby = test_set.groupby(train_set.columns[0])
    valid_count = int(test_groupby.size().mean())

    # split test data set 
    test_numbers, test_letters = _split_data_set(test_set)
    test_letters = _update_letter_labels(test_letters,updated_letter_mappings)
    test_numbers.to_csv(TESTING_DATA_PATH_NUMBERS, header=False, index=False)
    test_letters.to_csv(TESTING_DATA_PATH_LETTERS, header=False, index=False)

    # split train data set
    train_numbers, train_letters = _split_data_set(train_set)
    
    # split numbers into train and valid data sets
    train_numbers, valid_numbers = _train_valid_split(train_numbers, valid_count)
    train_numbers.to_csv(TRAINING_DATA_PATH_NUMBERS, header=False, index = False)
    valid_numbers.to_csv(VALIDATE_DATA_PATH_NUMBERS, header=False, index = False)

    # split letters into train and valid data sets
    train_letters, valid_letters = _train_valid_split(train_letters, valid_count)

    # Make balanced validation set before updating letter labels
    pd.concat([valid_numbers, valid_letters]).to_csv(VALIDATION_DATA_PATH_CSV, header=False, index = False)

    train_letters = _update_letter_labels(train_letters,updated_letter_mappings)
    valid_letters = _update_letter_labels(valid_letters,updated_letter_mappings)
    train_letters.to_csv(TRAINING_DATA_PATH_LETTERS, header=False, index = False)
    valid_letters.to_csv(VALIDATE_DATA_PATH_LETTERS, header=False, index = False)

    