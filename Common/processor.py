import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

class Processor():
    def __init__(self, model, optimizer, criterion):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._train_accu, self._train_loss = [], []
        self._valid_accu, self._valid_loss = [], []
        
    def training(self, dataloader):
        # switch to train mode
        self._model.train()
        total_accu, total_loss, total_count = 0, 0, 0

        for label, text, seq_lengths in dataloader:
            self._optimizer.zero_grad()

            predicted_label = self._model(text, seq_lengths)
            loss = self._criterion(predicted_label, label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
            self._optimizer.step()

            total_accu += (predicted_label.argmax(1) == label).sum().item()
            total_loss += loss.item() * label.size(0)
            total_count += label.size(0)

        avg_accu = total_accu/total_count
        avg_loss = total_loss/total_count

        self._train_accu.append(avg_accu)
        self._train_loss.append(avg_loss)

        return avg_accu, avg_loss

    def validation(self, dataloader, type="valid"):
        # switch to evaluate mode
        self._model.eval()
        total_accu, total_loss, total_count = 0, 0, 0

        with torch.no_grad():
            for label, text, seq_lengths in dataloader:
                # compute output
                predicted_label = self._model(text, seq_lengths)
                loss = self._criterion(predicted_label, label)

                # measure accuracy and record loss
                total_accu += (predicted_label.argmax(1) == label).sum().item()
                total_loss += loss.item() * label.size(0)
                total_count += label.size(0)

        avg_accu = total_accu/total_count
        avg_loss = total_loss/total_count
        
        if type == "valid":
            self._valid_accu.append(avg_accu)
            self._valid_loss.append(avg_loss)

        return avg_accu, avg_loss

    def prediction(self, dataloader):
        self._model.eval()
        pred_values, true_values = [], []

        with torch.no_grad():
            for labels, text, seq_lengths in dataloader:
                # Predict the labels
                predicted_label = self._model(text, seq_lengths)
                _, predicted = torch.max(predicted_label.data, 1)

                for pred, true in zip(predicted, labels):
                    pred_values.append(pred.item())
                    true_values.append(true.item())

        return true_values, pred_values

    def evaluation(self, true_values, pred_values):
        self._loss_accu_plot(self._train_loss, self._valid_loss, "Loss")
        self._loss_accu_plot(self._train_accu, self._valid_accu, "Accuracy")
        self._conf_matrix_plot(true_values, pred_values)
        precision, recall, f1score = self._metrics_plots(true_values, pred_values)

        return precision, recall, f1score
    
    def _loss_accu_plot(self, train, valid, type):
        plt.figure()
        plt.plot(range(1, len(train)+ 1), train, label=f"Training {type}", marker="*")
        plt.plot(range(1, len(valid)+ 1), valid, label=f"Validation {type}", marker="*")
        plt.title(f"Training and Validation {type} Against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(f"{type}")
        plt.grid()
        plt.legend()
        plt.savefig(f"./A/Plots/{type} Against Epoch.PNG")

    def _conf_matrix_plot(self, true, pred):
        conf_matrix = ConfusionMatrixDisplay(confusion_matrix(true, pred))
        conf_matrix.plot()
        conf_matrix.figure_.savefig("./A/Plots/Confusion Matrix.PNG")
    
    def _metrics_plots(self, true, pred):
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
        plt.savefig("./A/Plots/Performance Metrics.PNG")

        return precision, recall, f1score