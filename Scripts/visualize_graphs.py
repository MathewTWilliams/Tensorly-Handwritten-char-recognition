# Author: Matt Williams
# Version: 5/5/2022
# This is a 'playground' scrip that was used to make visualizations based on result files.
import matplotlib.pyplot as plt
from constants import PYT_RESULTS_FOLDER, TF_RESULTS_FOLDER
from save_results import load_cnn_results

train_loss = "Train Loss per Epoch"
class_report = "Classification Report"
train_time = "Training Time"
valid_loss = "Valid Loss per Epoch"
accuracy = "accuracy"
weighted_avg = "weighted avg"
precision = "precision"
recall = "recall"
f1_score = "f1-score"
num_predictions = "# Predictions"
prediction_time = "Prediction Time"
name = "Name"
dataset = "Dataset"
back_prop = "Back Propogation Time"




if __name__ == "__main__": 

    pytorch_results = load_cnn_results(PYT_RESULTS_FOLDER)
    tensorflow_results = load_cnn_results(TF_RESULTS_FOLDER)

    cur_dataset = "Balanced"
    cur_value = f1_score
    cur_results = [result for result in tensorflow_results if result[dataset] == cur_dataset]

    #for result in cur_results: 
        #plt.plot(result[cur_value], label = result[name])




    values = []
    names = []

    for result in cur_results:
        values.append(result[class_report][weighted_avg][cur_value])
        names.append(result[name])

    print(names)
    print(values)

    '''x_pos = [ i for i,_ in enumerate(cur_results)]


    plt.bar(x_pos, values)
    plt.xticks(x_pos, names)

   

    plt.title("Letters Back Propogation Times", fontsize = 15)
    plt.xlabel("Names", fontsize = 14)
    plt.ylabel("Back Propogation Time (including validation)", fontsize=14)
    #ax = plt.gca()
    #ax.yaxis.get_major_locator().set_params(integer = True)
    #ax.xaxis.get_major_locator().set_params(integer = True)
    #plt.ylim(0.85,1)
    plt.legend()
    plt.show()'''