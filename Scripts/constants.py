# Author: Matt Williams
# Version: 2/21/2022


from pathlib import PurePath, Path

# File path for the training data
TRAINING_DATA_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-train.csv").as_posix()

# File path for the test data
TESTING_DATA_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-test.csv").as_posix()
                        
# File path for the label mappings
EMNIST_MAPPING_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-mapping.txt").as_posix()


PIXEL_SIDE_LENGTH = 28
                        