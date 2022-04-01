import json
import os
from constants import *
from pathlib import PurePath


def save_cnn_results(results_dict, folder): 
    if not os.path.exists(folder): 
        os.mkdir(folder)

    file_name = "results_" + str(len(os.listdir(folder)) + 1) + ".json"

    path = PurePath.joinpath(folder, file_name).as_posix()

    with open(path, "w+", encoding='utf-8') as file: 
        json.dump(results_dict, file, indent = 1)

def load_cnn_results(folder, name):

    result = {}
    path = PurePath.joinpath(folder, name).as_posix()

    if not os.path.exists(path): 
        return None

    with open(path, "r+", encoding="utf-8") as file: 
        result = json.load(file)
    
    return result
