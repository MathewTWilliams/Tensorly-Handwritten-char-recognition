# Author: Matt Williams
# Version: 3/27/2022


from pathlib import PurePath, Path

# File path for the complete training data in csv form
# Not to be used outside of edit_dataset.py
TRAINING_DATA_PATH_CSV = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-train.csv").as_posix()

# File path for the complete test data in csv form
# Not to be used outside of edit_dataset.py
TESTING_DATA_PATH_CSV = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-test.csv").as_posix()

VALIDATION_DATA_PATH_CSV = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-valid.csv").as_posix()

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

                        
# File path for the label mappings (original)
EMNIST_MAPPING_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-mapping.txt").as_posix()

# File path for the updated letter mappings
UPDATED_LET_MAPPINGS_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                                "Dataset/updated-letter-mappings.txt").as_posix()


TF_RESULTS_FOLDER = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Results/Tensorflow/")

PYT_RESULTS_FOLDER = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Results/Py-Torch/")

PIXEL_SIDE_LENGTH = 28

LABEL_COL = 0

PYT_INPUT_SIZE = (1, PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH)

BATCH_SIZE = 32

N_EPOCHS = 20

N_NUM_CLASSES = 10
N_LET_CLASSES = 37
N_BAL_CLASSES = N_NUM_CLASSES + N_LET_CLASSES

VALIDATE = True