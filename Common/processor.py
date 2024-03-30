import torch

class Processor():
    '''
    A class for training, evaluation and predicting Pytorch model.

    Parameters:
        model (torch.nn.Module): 
            The Pytorch model.
        optimizer (torch.optim):
            The selected optimizer.
        criterion (torch.nn):
            The selected criterion.

    Methods:
        training(dataloader):
            The training loop.
        validation(dataloader, type="valid")
            The validation loop.
        prediction(dataloader)
            The prediction loop.
        get_evaluation_info(self)
            Return all evaluation information.
    '''
    def __init__(self, model, optimizer, criterion):
        '''
        Initiate the processor. 

        Parameters:
            model (torch.nn.Module): 
                The Pytorch model.
            optimizer (torch.optim):
                The selected optimizer.
            criterion (torch.nn):
                The selected criterion.
        '''
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._train_accu, self._train_loss = [], []
        self._valid_accu, self._valid_loss = [], []

    def training(self, dataloader):
        '''
        The training loop.

        Parameter:
            dataloader (DataLoader):
                The dataloader containing the train set. 

        Return:
            avg_accu (list):
                The average accuracy of this epoch.
            avg_loss (list):
                The average loss of this epoch.
        '''
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
        '''
        The validation loop.

        Parameters:
            dataloader (list):
                The dataloader containing the validation/ test set.
            type (str):
                Determine whether the function is used for validation or test set. 
        
        Return:
            avg_accu (list):
                The average accuracy.
            avg_loss (list):
                The average loss.
        
        '''
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
        else:
            print("-" * 45)
            print(f"| Test accuracy:  {avg_accu:5.3f} | test loss:  {avg_loss:5.3f} |")
            print("-" * 45)
        return avg_accu, avg_loss

    def prediction(self, dataloader):
        '''
        The prediction loop.

        Parameters:
            dataloader (list):
                The dataloader containing the test set.
        
        Return:
            true_values (list):
                The true labels of the test set.
            pred_values (list):
                The predicted labels of the test set.
        '''
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

        self._true_values = true_values
        self._pred_values = pred_values

        return true_values, pred_values

    def get_evaluation_info(self):
        '''
        Return all evaluation information.

        Returns:
            self._train_accu (list):
                All training accuracies.
            self._train_loss (list):
                All training losses.
            self._valid_accu (list):
                All validation accuracies.
            self._valid_loss (list):
                All validation losses.
            self._true_values (list):
                True labels of the test set. 
            self._pred_values (list):
                Predicted labels of the test set. 
        '''
        return self._train_accu, self._train_loss, self._valid_accu, self._valid_loss, self._true_values, self._pred_values
