import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

class EvaluationPlots():
    '''
    A class for plotting the results of different models.

    Parameter:
        type (str):
            The type of the model (RNN_LSTM, RNN_GRU, NN_Normal, Transformer).
    
    Methods:
        get_all_plots(train_accu, train_loss, valid_accu, valid_loss, true_values, pred_values):
            Plot all figures. 
    '''
    def __init__(self, type="RNN_LSTM"):
        '''
        Initiate the class.

        Parameter:
            type (str):
                The type of the model (RNN_LSTM, RNN_GRU, NN_Normal, Transformer).
        '''
        self._type = type

    def get_all_plots(self, train_accu, train_loss, valid_accu, valid_loss, true_values, pred_values):
        '''
        Plot all figures. 

        Parameters:
            train_accu (list):
                Training accuracy of past epochs.
            train_loss (list):
                Training loss of past epochs.
            valid_accu (list):
                Validation accuracy of past epochs.
            valid_loss (list):
                Validation loss of past epochs.
            true_values (list): 
                True labels of the test dataset.
            pred_values (list):
                Predicted labels of the test dataset.
        '''
        if self._type != "Transformer":
            epochs = range(1, len(train_accu) + 1)
        else:
            epochs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        self._loss_accu_plot(train_loss, valid_loss, epochs, "Loss")
        self._loss_accu_plot(train_accu, valid_accu, epochs, "Accuracy")
        self._conf_matrix_plot(true_values, pred_values)
        self._metrics_plots(true_values, pred_values)
    
    def _loss_accu_plot(self, train, valid, epoch, type):
        '''
        Plot the loss against accuracy plot for training and validation.

        Parameter:
            train (list):
                Training accuracy/ loss.
            valid (list):
                Validation accuracy/ loss.
            epoch (list):
                The epochs in list.
            type (str):
                The type of model.
        '''
        plt.figure()
        plt.plot(epoch, train, label=f"Training {type}", marker="*")
        plt.plot(epoch, valid, label=f"Validation {type}", marker="*")
        plt.title(f"Training and Validation {type} Against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(f"{type}")
        plt.grid()
        plt.legend()
        if self._type != "Transformer":
            save_dir = f"A/Plots/{self._type}/{type} Against Epoch.PNG"
        else:
            save_dir = f"B/Plots/{type} Against Epoch.PNG"
        plt.savefig(save_dir)

    def _conf_matrix_plot(self, true, pred):
        '''
        Plot the confusion matrix. 

        Parameter:
            true (list):
                True labels of the test dataset.
            pred (list)
                Predicted labels of the test dataset.
        '''
        conf_matrix = ConfusionMatrixDisplay(confusion_matrix(true, pred))
        conf_matrix.plot()
        if self._type != "Transformer":
            save_dir = f"A/Plots/{self._type}/Confusion Matrix.PNG"
        else:
            save_dir = "B/Plots/Confusion Matrix.PNG"
        conf_matrix.figure_.savefig(save_dir)
    
    def _metrics_plots(self, true, pred):
        '''
        Plot the bar diagrams for precision, recall and f1score.

        Parameter:
            true (list):
                True labels of the test dataset.
            pred (list)
                Predicted labels of the test dataset.
        '''
        micro_precision, micro_recall, micro_f1score, _ = precision_recall_fscore_support(true, pred, average='micro')
        macro_precision, macro_recall, macro_f1score, _ = precision_recall_fscore_support(true, pred, average='macro')
        class_precision, class_recall, class_f1score, _ = precision_recall_fscore_support(true, pred, average=None)
        
        precision = np.append(class_precision, np.append(micro_precision, macro_precision))
        recall = np.append(class_recall, np.append(micro_recall, macro_recall))
        f1score = np.append(class_f1score, np.append(micro_f1score, macro_f1score))

        precision = [float(format(100 * i, '.3f')) for i in precision]
        recall = [float(format(100 * i, '.3f')) for i in recall]
        f1score = [float(format(100 * i, '.3f')) for i in f1score]

        classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Micro", "Macro"]
        ticks = np.arange(len(classes))
        plt.figure()
        plt.bar(ticks, precision, 0.2, label = "Precision")
        plt.bar(ticks + 0.2, recall, 0.2, label = "Recall")
        plt.bar(ticks + 0.4, f1score, 0.2, label = "F1")
        plt.xlabel("Classes")
        plt.ylabel("Score")
        plt.ylim(0 ,120)
        plt.grid()
        plt.title("Performace Metrics")
        plt.xticks(ticks + 0.2, classes)
        plt.legend()
        if self._type != "Transformer":
            save_dir = f"A/Plots/{self._type}/Performance Metrics.PNG"
        else:
            save_dir = "B/Plots/Performance Metrics.PNG"
        plt.savefig(save_dir)

    