import torch.nn as nn

class NN(nn.Module):
    '''
    A class for a simple NN model. 

    Parameters:
        vocab_size (int):
            The number of vocabularies.
        embed_dim (int):
            The size of the embedded layer.
        num_class (int):
            The total number of classes.
    
    Method:
        init_weights():
            Setup the weights of the embedding and fully connected layers.
        forward(text, offsets):
            Forward propagation.
    '''
    def __init__(self, vocab_size, embed_dim, num_class):
        '''
        Initiate the NN model.

        Parameters:
        vocab_size (int):
            The number of vocabularies.
        embed_dim (int):
            The size of the embedded layer.
        num_class (int):
            The total number of classes.
        '''
        super(NN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        '''
        Setup the weights of the embedding and fully connected layers.
        '''
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        '''
        Forward propagation.

        Parameters:
            text (torch.tensor):
                The text for training.
            offsets (torch.tensor):
                The offsets in the text. 
        
        Return:
            self.fc(embedded) (torch.tensor):
                The predicted output. 
        '''
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
