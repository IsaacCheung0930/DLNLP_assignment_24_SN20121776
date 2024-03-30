class EarlyStopper():
    '''
    A class for early stopping when no improvement is observed. Credit to https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    Parameter:
        patience (int):
            The number of allowances before triggering early stopping.
        min_delta (float):
            The acceptable range of fluctuation. 

    Method:
        early_stop(validation_loss):
            The function to check for early stopping.
    '''
    def __init__(self, patience=1, min_delta=0):
        '''
        Initiate the checker. 

        Parameters:
        patience (int):
            The number of allowances before triggering early stopping.
        min_delta (float):
            The acceptable range of fluctuation. 

        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        '''
        Function for comparing current and previous validation loss.

        Parameter:
            validation_loss (float): 
                Validation loss from current epoch.
        Return:
            True/ False (bool):
                Whether early stopping is triggered.
        '''
        # If the current loss is smaller, reset the counter.
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # Increase the counter if requirement satisfied.
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False