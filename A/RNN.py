import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, model="LSTM",
                 dropout=0.3, hidden_size=32, num_layers=2, use_last=False):
        super(RNN, self).__init__()

        self.use_last = use_last
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

        if model == "LSTM":
            self.rnn = nn.LSTM(input_size=embed_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                dropout=dropout,
                                batch_first=True, 
                                bidirectional=False)
        else:
            self.rnn = nn.GRU(input_size=embed_size, 
                              hidden_size=hidden_size, 
                              num_layers=num_layers, 
                              dropout=dropout,
                              batch_first=True, 
                              bidirectional=True)
            
        self.batch_normalisation = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, seq_lengths):
        x_embed = self.embedding(data)
        x_embed = self.dropout(x_embed)
        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.rnn(packed_input, None)
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_rnn = self.relu(out_rnn)

        row_indices = torch.arange(0, data.size(0)).long()
        col_indices = seq_lengths - 1

        if self.use_last:
            last_tensor=out_rnn[row_indices, col_indices, :]
        else:
            last_tensor = out_rnn[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        full_connected_input = self.batch_normalisation(last_tensor)

        output = self.linear(full_connected_input)
        output = self.softmax(output)

        return output
