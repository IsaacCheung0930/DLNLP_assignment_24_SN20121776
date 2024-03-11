import torch
import torch.nn as nn
from preprocessing import Preprocess
from custom_dataloader import Custom_dataloader
import time

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def predict(model, text, pipeline):
        with torch.no_grad():
            text = torch.tensor(pipeline("text", text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess("Datasets/train.csv", read=True)
    data_list = preprocess.get_preprocessed_data()
    custom_dataloader = Custom_dataloader(data_list)
    train_dataloader, valid_dataloader, test_dataloader = custom_dataloader.get_dataloader(padding=False)

    num_class = len(set([label for (label, text) in data_list]))
    vocab_size = len(custom_dataloader.vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    # Hyperparameters
    EPOCHS = 10  # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader, model, optimizer, criterion, epoch)
        accu_val = evaluate(valid_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model, criterion)
    print("test accuracy {:8.3f}".format(accu_test))

    class_label = {1: "Society & Culture", 
                    2: "Science & Mathematics", 
                    3: "Health", 
                    4: "Education & Reference", 
                    5: "Computers & Internet", 
                    6: "Sports", 
                    7: "Business & Finance", 
                    8: "Entertainment & Music", 
                    9: "Family & Relationships", 
                    10: "Politics & Government"}


    ex_text_str = "make friendship click"

    model = model.to("cpu")

    print("This is a %s class" % class_label[predict(model, ex_text_str, custom_dataloader.pipelines)])

main()