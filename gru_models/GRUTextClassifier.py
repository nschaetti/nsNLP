#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the nsNLP toolbox.
# The RnsNLP toolbox is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

# Import packages
import numpy as np
import mdp
from datetime import datetime
from sys import getsizeof
from nsNLP.classifiers.TextClassifier import TextClassifier
import matplotlib.pyplot as plt
from decimal import *
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .modules.GRUModule import GRUModule


# GRU classifier model
class GRUTextClassifier(TextClassifier):
    """
    Gated Recurrent Units classifier model
    """

    # Constructor
    def __init__(self, classes, hidden_size, converter, n_layers=1, embedding_dim=300, learning_rate=0.1):
        """
        Constructor
        :param classes:
        :param hidden_size:
        :param converter:
        :param embedding_dim:
        :param learning_rate:
        """
        # Super
        super(GRUTextClassifier, self).__init__(classes=classes)

        # Properties
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._n_layers = n_layers

        # Create model
        self._model = GRUModule(embedding_dim, hidden_size, self._n_classes, n_layers)
        self._loss_function = nn.NLLLoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=learning_rate)
    # end __init__

    ##############################################
    # Override
    ##############################################

    # To string
    def __str__(self):
        """
        To string
        :return:
        """
        return "GRUTextClassifier(n_classes={}, embedding_size={}, hidden_size={}, n_layers={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, self._n_layers, getsizeof(self))
    # end __str__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"GRUTextClassifier(n_classes={}, embedding_size={}, hidden_size={}, n_layers={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, self._n_layers, getsizeof(self))
    # end __unicode__

    ##############################################
    # Private
    ##############################################

    # Train
    def _train(self, x, y, verbose=False):
        """
        Add a training example
        :param x: Text file example
        :param y: Corresponding author
        """
        self._examples.append((x, y))
    # end _train

    # Finalize the training phase
    def _finalize_training(self, verbose=False):
        """
        Finalize the training phase
        :param verbose: Verbosity
        """
        pass
    # end _finalize_training

    # Classify a text file
    def _classify(self, text):
        """
        Classify text
        :param text: Text to classify
        :return: Predicted class and class probabilities
        """
        pass
    # end _classify

    # Reset learning but keep reservoir
    def _reset_model(self):
        """
        Reset model learned parameters
        """
        pass
    # end _reset_model

    # Generate training data from text
    def _generate_training_data(self, text, author):
        """
        Generate training data from text file.
        :param text: Text
        :param author: Corresponding author.
        :return: Data set inputs
        """
        pass
    # end generate_training_data

    # Generate text data from text file
    def _generate_test_data(self, text):
        """
        Generate text data from text file
        :param text: Text
        :return: Test data set inputs
        """
        pass
    # end generate_text_data

# end LSTMTextClassifier
