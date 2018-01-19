
# Imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# A LSTM module
class LSTMModule(nn.Module):
    """
    A simple LSTM module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, voc_size=0, train_embeddings=False):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param train_embeddings:
        """
        super(LSTMModule, self).__init__()

        # Properties
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._train_embeddings = train_embeddings

        # Need to create embeddings layer
        if self._train_embeddings:
            self._word_embeddings = nn.Embedding(voc_size, input_dim)
        else:
            self._word_embeddings = None
        # end if

        # The LSTM takes workds as inputs with
        # dimensionality input_dim, and outputs hidden states
        # with dimensionality hidden_dim,
        # finally the output is the different classes with
        # dimensionality output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to class probabilities
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)

        # Initiate hidden state
        self.hidden = self._init_hidden()
    # end __init__

    #######################################
    # PUBLIC
    #######################################

    # Forward pass
    def forward(self, inputs):
        """
        Forward pass
        :param inputs:
        :return:
        """
        # Compute word embeddings if needed
        if self._train_embeddings:
            embeds = self._word_embeddings(inputs)
            lstm_out, self.hidden = self.lstm(embeds.view(len(inputs), 1, -1), self.hidden)
        else:
            lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        # end if

        # Compute class space
        class_space = self.hidden2tag(lstm_out.view(len(inputs), -1))

        # Compute class scores with softmax
        class_score = F.log_softmax(class_space)

        # Return class scores
        return class_score
    # end forward

    #######################################
    # Private
    #######################################

    # Init hidden layer
    def _init_hidden(self):
        """
        Init hidden layer
        :return:
        """
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    # end init_hidden

# end LSTMModule
