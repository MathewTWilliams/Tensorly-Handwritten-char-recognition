import json
import os
from constants import *
from pathlib import PurePath


def save_cnn_results(results_dict, folder): 
    '''Given a dictionary and a results folder to save to, save the dictionary to that folder.'''
    if not os.path.exists(folder): 
        os.mkdir(folder)

    file_name = "results_" + str(len(os.listdir(folder)) + 1) + ".json"

    path = PurePath.joinpath(folder, file_name).as_posix()

    with open(path, "w+", encoding='utf-8') as file: 
        json.dump(results_dict, file, indent = 1)

def load_cnn_results(folder):
    """Given a results folder load and return all the results dictionaries in that folder."""
    results = []
    directory = folder.as_posix()

    if not os.path.exists(directory): 
        return None

    for file in os.listdir(directory):
        path = PurePath.joinpath(folder, file).as_posix() 
        with open(path, "r+", encoding="utf-8") as file: 
            result = json.load(file)
            results.append(result)
    
    return results
