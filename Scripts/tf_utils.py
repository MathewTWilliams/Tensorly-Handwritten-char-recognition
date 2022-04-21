# Author: Matt Williams
# Version: 3/30/2022

from gc import callbacks
from sklearn.metrics import classification_report
import numpy as np
from save_results import save_cnn_results
from constants import *
from datetime import datetime

# Includes Pylance can't confirm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

#TODO edit method to include infer/run decomposition, compilation, run model method
def run_model(model, train_set_func, valid_set_func, test_set_func, \
    model_name, dataset_name, normalize = True, num_color_channels = 1):

    training_hist = None
    train_x, train_y = train_set_func(normalize=normalize, num_color_channels=num_color_channels)
    valid_x, valid_y = valid_set_func(normalize=normalize, num_color_channels=num_color_channels)
    test_x, test_y = test_set_func(normalize = normalize, num_color_channels = num_color_channels)

    one_hot_train_y = to_categorical(train_y)
    one_hot_valid_y = to_categorical(valid_y)
    one_hot_test_y = to_categorical(test_y)

    stop = EarlyStopping(monitor = "val_loss", mode = "min")

    start = datetime.now()
    if VALIDATE:
         training_hist = model.fit(train_x, one_hot_train_y, epochs = N_EPOCHS, batch_size = BATCH_SIZE, \
             validation_data = (valid_x, one_hot_valid_y), verbose = 0, callbacks=[stop])
    else: 
         training_hist = model.fit(train_x, one_hot_train_y, epochs = N_EPOCHS, batch_size = BATCH_SIZE, verbose = 0, callbacks=[stop])
    end = datetime.now()

    predict_start = datetime.now()
    one_hot_predictions = model.predict(test_x, batch_size = BATCH_SIZE)
    predict_end = datetime.now()
    predictions = np.argmax(one_hot_predictions, axis = -1)
    class_report = classification_report(test_y, predictions, output_dict=True)
    class_report['# Predictions'] = len(test_y)
    class_report["Prediction Time"] = (predict_end-predict_start).total_seconds()

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


def compile_model(model): 
    opt = SGD(learning_rate = 0.01, momentum = 0.9)
    model.compile(optimizer = opt, loss = "categorical_crossentropy")

    return model


