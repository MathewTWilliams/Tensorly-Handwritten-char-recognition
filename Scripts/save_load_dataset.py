# Author: Matt Williams
# Version: 2/21/2022

import json
from constants import *
import os


def _load_json(path): 
    """ Method that takes in the file path of a .json file and returns the 
        python dictionary of that json file."""
    if not os.path.exists(path):
        return {}


    file_path_split = path.split(".")

    #the last element in the split should be the file extention
    if not file_path_split[-1] == "json": 
        return {}

    with open(path, "r+") as file: 
        json_dict = json.load(file)
        return json_dict


def save_json(json_obj, path):
    """Given a json object and a file path, store the json object at the given file path""" 

    if path.split('.')[-1] == "json":
        if os.path.exists(path): 
            os.remove(path)


        with open(path, "w+") as file: 
            json.dump(json_obj, file, indent=1)
    
    


def load_training_letter_dataset(): 
    return _load_json(TRAINING_DATA_PATH_LETTERS)

def load_training_number_dataset(): 
    return _load_json(TRAINING_DATA_PATH_NUMBERS)

def load_testing_letter_dataset(): 
    return _load_json(TESTING_DATA_PATH_LETTERS)

def load_testing_number_dataset(): 
    return _load_json(TESTING_DATA_PATH_NUMBERS)

def load_validation_letter_dataset(): 
    return _load_json(VALIDATE_DATA_PATH_LETTERS)

def load_validation_number_dataset(): 
    return _load_json(VALIDATE_DATA_PATH_NUMBERS)

# just for testing purposes
if __name__ == "__main__":
    print(load_testing_number_dataset()['0'])
