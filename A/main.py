import sys
import os
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from Common.preprocessor import Preprocessor
from Common.custom_dataloader import CustomDataloader
from Common.processor import Processor
from Common.early_stopper import EarlyStopper
from B.huggingface import Transformer
from RNN import RNN
from NN import TextClassificationModel

def main():
    # Enable CUDA usage
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using {device} device.")

    # Preprocessing
    preprocess = Preprocessor(data_dir="Datasets/raw/train.csv", 
                              read=True, 
                              sample_size=100000)
    data_list = preprocess.get_preprocessed_data()

    # Obtain dataloaders for the model
    custom_dataloader = CustomDataloader(data_list)
    #train_dataloader, val_dataloader, test_dataloader = custom_dataloader.get_dataloader(padding=True)
    train_df = custom_dataloader.get_split_data_df("train")
    val_df = custom_dataloader.get_split_data_df("val")
    test_df = custom_dataloader.get_split_data_df("test")
    transformer = Transformer(train_df, val_df, test_df)

    transformer.train()
    '''
    # Define model parameters
    vocab_size = len(custom_dataloader.vocab)
    num_class = len(preprocess.get_class_distribution())
    emsize = 128
    
    model = RNN(vocab_size=vocab_size, 
                embed_size=emsize, 
                num_classes=num_class, 
                model = "LSTM",
                use_last=True).to(device)
    
    #model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    early_stopper = EarlyStopper(patience=3, min_delta=0)
    process = Processor(model, optimizer, criterion)

    epoch = 20
    train_start_time = time.time()
    for epoch in range(1, epoch+1):
        epoch_start_time = time.time()
        train_accu, train_loss = process.training(train_dataloader)
        valid_accu, valid_loss = process.validation(val_dataloader, type="valid")

        print("-" * 45)
        print(f"| End of epoch: {epoch:7d} | Time: {(time.time() - epoch_start_time):10.2f}s |")
        print(f"| Train accuracy: {train_accu:5.3f} | Train loss: {train_loss:5.3f} |")
        print(f"| Valid accuracy: {valid_accu:5.3f} | Valid loss: {valid_loss:5.3f} |")
        print("-" * 45)

        if early_stopper.early_stop(valid_loss):
            print("Early stopping...")             
            break

    print(f"Total training time: {(time.time() - train_start_time):.2f}s")

    print("Evaluating entire test dataset...")
    test_accu, test_loss = process.validation(test_dataloader, type="test")

    print("-" * 45)
    print(f"| Test accuracy:  {test_accu:5.3f} | test loss:  {test_loss:5.3f} |")
    print("-" * 45)

    pred_values, true_values = process.prediction(test_dataloader)
    precision, recall, f1score = process.evaluation(pred_values, true_values)
    
    test_num = 10
    print(f"Randomly evaluating {test_num} test questions...")
    test_samples, sample_dataloader = custom_dataloader.get_test_samples(test_num)
    sample_pred, sample_true = process.prediction(sample_dataloader)
    print("-" * 45)
    for i in range(test_num):
        print("| Question |")
        print(f"{test_samples[i][2]}")
        print("| Answer   | ")
        print(f"{test_samples[i][3]}")
        print(f"| Predicted: {sample_pred[i] + 1} | True: {sample_true[i] + 1} | ")
    print("-" * 45)
    '''



main()