# Author: Matt Williams
# Version: 2/21/2022

from constants import EMNIST_MAPPING_PATH


def edit_label_mappings(): 

    new_lines = []
    
    with open(EMNIST_MAPPING_PATH, mode = "r", encoding = "utf-8") as file: 
        old_lines = file.readlines()

        for line in old_lines: 
            mapping = line.strip().split()
            mapping[1] = chr(int(mapping[1]))
            
            new_lines.append(mapping[0] + " " + mapping[1])
        

    with open(EMNIST_MAPPING_PATH, mode = "w", encoding = "utf-8") as file:

        for i, line in enumerate(new_lines): 
            file.write(line)
            if i < len(new_lines) - 1: 
                file.write("\n")

if __name__ == "__main__": 
    edit_label_mappings()
