import sys
import os
import time
import torch
import argparse
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

from A.RNN_model import RNN
from B.transformer_model import Transformer
from Common.preprocessor import Preprocessor
from Common.custom_dataloader import CustomDataloader
from Common.processor import Processor
from Common.early_stopper import EarlyStopper
from Common.evaluation_plots import EvaluationPlots

def main(args):
    '''
    Run RNN (LSTM, GRU), NN and Transformer based on the input args.

    Paramter:
        args (Namespace): 
            The arguments from argparse.
    '''
    # Check if selected model is valid
    if args.model not in ["RNN", "Transformer"]:
        print("Invalid selected model.")
        sys.exit(0)
    else:
        print(f"{args.model} model selected")

    # Enable CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print(f"GPU: {torch.cuda.get_device_name(device=device)}")

    # Preprocessing
    preprocess = Preprocessor(data_dir="Datasets/raw/train.csv", read=True, sample_size=100000)
    preprocess.get_class_distribution()
    preprocess.get_length_distribution()

    # Export text length and class distribution
    data_list = preprocess.get_preprocessed_data()
    custom_dataloader = CustomDataloader(data_list)
    
    # --------------------------- RNN MODEL ---------------------------------#
    if args.model == "RNN":
        # Obtain dataloaders for RNN model
        train_dataloader, val_dataloader, test_dataloader = custom_dataloader.get_dataloader(padding=True)

        # Define model parameters
        vocab_size = len(custom_dataloader.vocab)
        print(vocab_size)
        num_class = len(preprocess.get_class_distribution())
        emsize = 128
        
        model = RNN(vocab_size=vocab_size, 
                    embed_size=emsize, 
                    num_classes=num_class, 
                    model=args.type,
                    use_last=True).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        early_stopper = EarlyStopper(patience=3, min_delta=0)
        process = Processor(model, optimizer, criterion)

        epoch = 20
        train_start_time = time.time()

        # Training
        print(f"Training {args.model} {args.type} model...")
        for epoch in range(1, epoch+1):
            epoch_start_time = time.time()
            train_accu, train_loss = process.training(train_dataloader)
            valid_accu, valid_loss = process.validation(val_dataloader, type="valid")

            print_train(epoch, epoch_start_time, train_accu, train_loss, valid_accu, valid_loss)

            if early_stopper.early_stop(valid_loss):
                print("Early stopping...")             
                break

        print(f"Total training time: {(time.time() - train_start_time):.2f}s")

        # Evaluation
        print("Evaluating entire test dataset...")
        test_accu, test_loss = process.validation(test_dataloader, type="test")
        print_test(test_accu, test_loss)

        # Prediction
        process.prediction(test_dataloader)

        # Plotting results
        NN_evaluation = EvaluationPlots(type= "_".join([args.model, args.type]))
        train_accu, train_loss, valid_accu, valid_loss, true_values, pred_values = process.get_evaluation_info()
        NN_evaluation.get_all_plots(train_accu, train_loss, valid_accu, valid_loss, true_values, pred_values)

    # -------------------------- TRANSFORMER MODEL -------------------------------#
    if args.model == "Transformer":
        # Obtain dataframes for the transformer model
        train_df, val_df, test_df = custom_dataloader.get_split_data_df()

        # Setup transformer model
        transformer = Transformer(train_df, val_df, test_df)
        
        # Training, evaluation and prediction
        transformer.training()
        transformer.evaluation()
        transformer.prediction()

        # Plotting results
        transformer_evaluation = EvaluationPlots(type="Transformer")
        train_accu, train_loss, valid_accu, valid_loss, true_values, pred_values = transformer.get_evaluation_info()
        transformer_evaluation.get_all_plots(train_accu, train_loss, valid_accu, valid_loss, true_values, pred_values)

def print_train(epoch, epoch_start_time, train_accu, train_loss, valid_accu, valid_loss):
    '''
    Print the training results.

    Parameters:
        epoch (int)
            Current number of epoch.
        epoch_start_time (float)
            Start time of current epoch.
        train_accu (float)
            Training accuracy of current epoch.
        train_loss (float)
            Training loss of current epoch.
        valid_accu (float)
            Validation accuracy of current epoch.
        valid_loss (float)
            Validation loss of current epoch.
    '''
    print("-" * 45)
    print(f"| End of epoch: {epoch:7d} | Time: {(time.time() - epoch_start_time):10.2f}s |")
    print(f"| Train accuracy: {train_accu:5.3f} | Train loss: {train_loss:5.3f} |")
    print(f"| Valid accuracy: {valid_accu:5.3f} | Valid loss: {valid_loss:5.3f} |")
    print("-" * 45)

def print_test(test_accu, test_loss):
    '''
    Print the test results.
    
    Parameters:
        test_accu (float):
            Accuracy of the test dataset.
        test_loss (float):
            Loss of the test dataset.
    '''
    print("-" * 45)
    print(f"| Test accuracy:  {test_accu:5.3f} | Test loss:  {test_loss:5.3f} |")
    print("-" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("-m", "--model", type=str, default="RNN", help="RNN, Transformer")
    parser.add_argument("-t", "--type", type=str, default="LSTM", help="LSTM, GRU")
    args = parser.parse_args()
    main(args)
