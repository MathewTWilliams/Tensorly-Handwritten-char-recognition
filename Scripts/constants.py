# Author: Matt Williams
# Version: 3/12/2022


from pathlib import PurePath, Path

# File path for the complete training data in csv form
# Not to be used outside of edit_dataset.py
TRAINING_DATA_PATH_CSV = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-train.csv").as_posix()

# File path for the complete test data in csv form
# Not to be used outside of edit_dataset.py
TESTING_DATA_PATH_CSV = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-test.csv").as_posix()


# File path for the letters training data in json form
TRAINING_DATA_PATH_LETTERS = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-train-letters.csv").as_posix()

# File path for the numbers training data in json form
TRAINING_DATA_PATH_NUMBERS = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-train-numbers.csv").as_posix()


# File path for the letters test data in json form
TESTING_DATA_PATH_LETTERS = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-test-letters.csv").as_posix()

# File path for the numbers test data in json form
TESTING_DATA_PATH_NUMBERS = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-test-numbers.csv").as_posix()


# File path for the letters validation data in json form
VALIDATE_DATA_PATH_LETTERS = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-validation-letters.csv").as_posix()

# File path for the letters validation data in json form
VALIDATE_DATA_PATH_NUMBERS = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-validation-numbers.csv").as_posix()

                        
# File path for the label mappings
EMNIST_MAPPING_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-mapping.txt").as_posix()


PIXEL_SIDE_LENGTH = 28

LABEL_COL = 0
                        