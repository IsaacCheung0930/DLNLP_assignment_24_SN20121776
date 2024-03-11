from preprocessing import Preprocess
from custom_dataloader import Custom_dataloader
from process import Process
from early_stopper import Early_stopper
from RNN import RNN

import time
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    preprocess = Preprocess(data_dir="Datasets/train.csv", 
                            read=True, 
                            sample_size=100000, 
                            include_content=False)
    data_list = preprocess.get_preprocessed_data()

    custom_dataloader = Custom_dataloader(data_list)
    train_dataloader, val_dataloader, test_dataloader = custom_dataloader.get_dataloader(padding=True)

    vocab_size = len(custom_dataloader.vocab)
    num_class = len(set([label for (label, _) in data_list]))
    emsize = 128

    model = RNN(vocab_size=vocab_size, 
                embed_size=emsize, 
                num_classes=num_class, 
                model = "LSTM",
                use_last=True).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    early_stopper = Early_stopper(patience=3, min_delta=0)
    process = Process(model, optimizer, criterion)

    epoch = 20
    train_start_time = time.time()
    for epoch in range(1, epoch+1):
        epoch_start_time = time.time()
        train_accu, train_loss = process.training(train_dataloader)
        valid_accu, valid_loss = process.validation(val_dataloader)

        print("-" * 45)
        print(f"| End of epoch: {epoch:7d} | Time: {(time.time() - epoch_start_time):10.2f}s |")
        print(f"| Train accuracy: {train_accu:5.3f} | Train loss: {train_loss:5.3f} |")
        print(f"| Valid accuracy: {valid_accu:5.3f} | Valid loss: {valid_loss:5.3f} |")
        print("-" * 45)

        if early_stopper.early_stop(valid_loss):
            print("Early stopping...")             
            break

    print(f"Total training time: {(time.time() - train_start_time):.2f}s")

    print("Evaluating test dataset...")
    test_accu, test_loss = process.validation(test_dataloader)

    print("-" * 45)
    print(f"| Test accuracy:  {test_accu:5.3f} | test loss:  {test_loss:5.3f} |")
    print("-" * 45)

    pred_values, true_values = process.prediction(test_dataloader)
    precision, recall, f1score = process.evaluation(pred_values, true_values)


main()