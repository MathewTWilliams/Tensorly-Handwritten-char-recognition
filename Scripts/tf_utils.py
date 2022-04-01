# Author: Matt Williams
# Version: 3/30/2022

from sklearn.metrics import classification_report
import numpy as np
from save_results import save_cnn_results
from constants import *
from datetime import datetime

# Includes Pylance can't confirm
from tensorflow.keras.utils import to_categorical


def run_model(model, train_set_func, valid_set_func, model_name, dataset_name, normalize = True, num_color_channels = 1):

    training_hist = None
    train_x, train_y = train_set_func(normalize=True, num_color_channels=1)
    valid_x, valid_y = valid_set_func(normalize=True, num_color_channels=1)

    one_hot_train_y = to_categorical(train_y)
    one_hot_valid_y = to_categorical(valid_y)

    start = datetime.now()
    if VALIDATE:
         training_hist = model.fit(train_x, one_hot_train_y, epochs = N_EPOCHS, batch_size = BATCH_SIZE, validation_data = (valid_x, one_hot_valid_y), verbose = 0)
    else: 
         training_hist = model.fit(train_x, one_hot_train_y, epochs = N_EPOCHS, batch_size = BATCH_SIZE, verbose = 0)
    end = datetime.now()

    one_hot_predictions = model.predict(valid_x, batch_size = BATCH_SIZE)
    predictions = np.argmax(one_hot_predictions, axis = -1)

    class_report = classification_report(valid_y, predictions, output_dict=True)

    model_summary = []
    model.summary(print_fn= lambda x: model_summary.append(x))


    results_dict = {
        "Name": model_name, 
        "Dataset": dataset_name,
        "Model": model_summary,
        "Train Loss per Epoch" : training_hist.history['loss'],
        "Classification Report": class_report, 
        "Training Time": (end - start).total_seconds(),
    }

    if VALIDATE:
        results_dict['Valid Loss Per Epoch'] = training_hist.history['val_loss']

    save_cnn_results(results_dict, TF_RESULTS_FOLDER)


