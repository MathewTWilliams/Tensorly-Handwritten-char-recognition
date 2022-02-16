# Author: Matt Williams
# Version: 2/15/2022


from pathlib import PurePath, Path

TRAINING_DATA_SET = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-train.csv").as_posix()

TESTING_DATA_SET = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                        "Dataset/emnist-balanced-test.csv").as_posix()


                        