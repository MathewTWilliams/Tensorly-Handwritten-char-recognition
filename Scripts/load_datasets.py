# Author: Matt Williams
# Version: 2/15/2022

import pandas as pd
from constants import TESTING_DATA_SET, TRAINING_DATA_SET


def _load_dataset(path): 
    dataset = pd.read_csv(path)
    return dataset


def load_training_dataset(): 
    return _load_dataset(TRAINING_DATA_SET)

def load_testing_dataset(): 
    return _load_dataset(TESTING_DATA_SET)



if __name__ == "__main__":
    load_testing_dataset()
    load_training_dataset()
