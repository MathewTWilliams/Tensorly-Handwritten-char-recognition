# Author: Matt Williams
# Version: 2/15/2022

import pandas as pd
from constants import *


def _load_dataset(path):
    dataset = pd.read_csv(path, header=None)

    label_dict = {}
    for row in dataset.values: 
        label = row[0]

        if label not in label_dict.keys(): 
            label_dict[label] = []
            
        label_dict[label].append(row[1:])

    return label_dict
    
def load_training_dataset(): 
    return _load_dataset(TRAINING_DATA_PATH)

def load_testing_dataset(): 
    return _load_dataset(TESTING_DATA_PATH)

# just for testing purposes
if __name__ == "__main__":
    load_testing_dataset()
