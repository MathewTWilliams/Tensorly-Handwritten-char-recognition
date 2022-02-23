# Author: Matt Williams
# Version: 2/22/2022

from constants import *
import pandas as pd

def get_mappings(): 

    mappings = {}

    with open(EMNIST_MAPPING_PATH, mode = "r", encoding = "utf-8") as file: 
        lines = file.readlines()
        for line in lines: 
            mapping = line.strip().split()
            mappings[int(mapping[0])] = chr(int(mapping[1]))

    return mappings


def edit_labels(): 

    mappings = get_mappings()
    label_col = 0

    train_data = pd.read_csv(TRAINING_DATA_PATH, header=None)
    test_data = pd.read_csv(TESTING_DATA_PATH, header=None)
    
    for o_label, n_label in mappings.items(): 
        train_data[label_col] = train_data[label_col].replace(o_label, n_label)
        test_data[label_col] = train_data[label_col].replace(o_label, n_label)

    train_data.to_csv(TRAINING_DATA_PATH, header=None, index = False)
    test_data.to_csv(TESTING_DATA_PATH, header=None, index=False)





if __name__ == "__main__": 
    edit_labels()
