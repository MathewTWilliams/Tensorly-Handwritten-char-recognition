import matplotlib.pyplot as plt
from constants import PYT_RESULTS_FOLDER, TF_RESULTS_FOLDER
from save_results import load_cnn_results

if __name__ == "__main__": 
    
    n_epochs = 20
    #keys

    train_loss = "Train Loss per Epoch"
    class_report = "Classification Report"
    train_time = "Training Time"
    valid_loss = "Valid Loss per Epoch"

    accuracy = "accuracy"
    weighted_avg = "weighted avg"
    precision = "precision"
    recall = "recall"

    f1_score = "f1-score"



    pyt_lenet = load_cnn_results(PYT_RESULTS_FOLDER, "results_1.json")

    pyt_alexnet = load_cnn_results(PYT_RESULTS_FOLDER, "results_2.json")

    tf_lenet = load_cnn_results(TF_RESULTS_FOLDER, "results_1.json")

    tf_alexnet = load_cnn_results(TF_RESULTS_FOLDER, "results_2.json")


    names = ["PYT LeNet-5", "PYT AlexNet", "Tf LeNet-5", "Tf AlexNet"]
    values = [
        pyt_lenet[class_report][weighted_avg][f1_score],
        pyt_alexnet[class_report][weighted_avg][f1_score], 
        tf_lenet[class_report][weighted_avg][f1_score],
        tf_alexnet[class_report][weighted_avg][f1_score]
    ]

    x_pos = [ i for i,_ in enumerate(names)]


    plt.bar(x_pos, values)
    plt.xticks(x_pos, names)
    #plt.plot(pyt_lenet[train_loss], label = "Lenet-5 Pytorch Training")
    #plt.plot(pyt_lenet[valid_loss], label = "Lenet-5 Pytorch Validation")
    #plt.plot(tf_lenet[train_loss], label = "Lenet-5 Tensorflow Training")
    #plt.plot(tf_lenet[valid_loss], label = "Lenet-5 Tensorflow Validation")
   

    plt.title("Validation F1-Score", fontsize = 15)
    plt.xlabel("Names", fontsize = 14)
    plt.ylabel("F1-Score", fontsize=14)
    ax = plt.gca()
    #ax.yaxis.get_major_locator().set_params(integer = True)
    #ax.xaxis.get_major_locator().set_params(integer = True)
    plt.ylim(0.85,1)
    plt.legend()
    plt.show()