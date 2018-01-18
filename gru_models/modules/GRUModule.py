
# Imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# A GRU module
class GRUModule(nn.Module):
    """
    A simple GRU module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, train_embeddings=False):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param num_layers:
        """
        super(GRUModule, self).__init__()

        # Properties
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
        self._linear = nn.Linear(hidden_dim, output_dim)
    # end __init__

    #####################################
    # PUBLIC
    #####################################

    # Forward pass
    def forward(self, inputs, hidden):
        """
        Forward pass
        :param inputs:
        :param hidden:
        :return:
        """
        seq_len = len(inputs)
        x, hidden = self._gru(inputs, hidden)
        x = x.select(1, seq_len-1).contiguous()
        x = x.view(-1, self._hidden_dim)
        x = self._linear(x)
        return x, hidden
    # end forward

    # Init hidden layer
    def init_hidden(self, N):
        weight = next(self.parameters()).data
        return autograd.Variable(weight.new(1, batch_size, self._hidden_dim).zero_())
    # end init_hidden
#

# end GRUModule
