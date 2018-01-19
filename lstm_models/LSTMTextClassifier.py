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
from sys import getsizeof
from nsNLP.classifiers.TextClassifier import TextClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from .modules.LSTMModule import LSTMModule
from decimal import Decimal


# LSTM classifier model
class LSTMTextClassifier(TextClassifier):
    """
    LSTM classifier model
    """

    # Properties
    _model = None
    _loss_function = None
    _optimizer = None
    _last_y = None

    # Constructor
    def __init__(self, classes, hidden_size, converter, embedding_dim=300, learning_rate=0.1, voc_size=0, n_epoch=300, aggregation='average', smoothing=0.001):
        """
        Constructor
        :param classes:
        :param hidden_size:
        :param converter:
        :param embedding_dim:
        :param learning_rate:
        """
        # Super
        super(LSTMTextClassifier, self).__init__(classes=classes)

        # Properties
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._n_epoch = n_epoch
        self._converter = converter
        self._voc_size = voc_size
        self._learning_rate = learning_rate
        self._aggregation = aggregation
        self._smoothing = smoothing

        # Create model
        self._reset_model()
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
        return "LSTMTextClassifier(n_classes={}, embedding_size={}, hidden_size={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, getsizeof(self))
    # end __str__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"LSTMTextClassifier(n_classes={}, embedding_size={}, hidden_size={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, getsizeof(self))
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
        # For each epoch
        for epoch in range(self._n_epoch):
            # For each training text file
            for index, (x, y) in enumerate(self._examples):
                # As pyTorch accumulates gradients we clear them
                # out before each instance.
                self._model.zero_grad()

                # Re-init the hidden state
                self._model.hidden = self._model.init_hidden()

                # Now we prepare the inputs and turn them into
                # variables.
                class_index = self._class_to_int(y)
                train_x, train_y = self._generate_training_data(x, class_index)

                # Run the forward pass
                tag_scores = self._model(x)

                # Compute loss, gradients and update parameters
                loss = self._loss_function(tag_scores, train_y)
                loss.backward()
                self._optimizer.step()
            # end for
        # end for
    # end _finalize_training

    # Classify a text file
    def _classify(self, text):
        """
        Classify text
        :param text: Text to classify
        :return: Predicted class and class probabilities
        """
        # Get inputs
        x = self._generate_test_data(text)

        # Get class scores
        class_scores = self._model(x)

        # Normalized
        y = class_scores - torch.min(class_scores)
        y /= torch.max(class_scores)

        # Save last y
        self._last_y = y

        # Get maximum probability class
        if self._aggregation == 'average':
            outputs_average = torch.mean(y, 0)
            _, best_class = torch.max(outputs_average, 0)
            return self._int_to_class(best_class), outputs_average
        elif self._aggregation == 'multiply':
            # Decimal score
            scores = list()
            for i in range(self._n_classes):
                scores.append(Decimal(1.0))
            # end for

            # For each outputs
            for pos in range(y.shape[0]):
                for i in range(self._n_classes):
                    if y[pos, i] == 0.0:
                        scores[i] = scores[i] * Decimal(self._smoothing)
                    else:
                        scores[i] = scores[i] * Decimal(y[pos, i])
                    # end if
                # end for
            # end for

            # Return the max
            max = 0.0
            max_c = None
            for i in range(self._n_classes):
                if scores[i] > max:
                    max_c = self._int_to_class(i)
                    max = scores[i]
                # end if
            # end for
            return max_c, scores
        elif self._aggregation == 'last':
            last_outputs = y[-1]
            _, best_class = torch.max(last_outputs, 0)
            return self._int_to_class(best_class), last_outputs
        # end if
    # end _classify

    # Reset learning but keep reservoir
    def _reset_model(self):
        """
        Reset model learned parameters
        """
        # Create model
        if self._converter is not None:
            self._model = LSTMModule(self._converter.get_n_inputs(), self._hidden_size, self._n_classes)
        else:
            self._model = LSTMModule(self._embedding_dim, self._hidden_size, self._n_classes, self._voc_size, train_embeddings=True)
        # end if

        # Loss function and optimizer
        self._loss_function = nn.NLLLoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._learning_rate)
    # end _reset_model

    # Generate training data from text
    def _generate_training_data(self, text, c):
        """
        Generate training data from text file.
        :param text: Text
        :param c: Corresponding class.
        :return: Data set inputs
        """
        # Get Temporal Representations
        reps = self._converter(text)

        # Converter type
        converter_type = type(self._converter)

        # Generate x and y
        x, _ = converter_type.generate_data_set_inputs(reps, self._n_classes, c)

        # Training length
        train_length = x.shape[0]

        # Targets as tensor
        y = torch.FloatTensor(train_length, self._n_classes).zero_()
        y[:, c] = 1

        # Inputs as Tensor
        x = torch.FloatTensor(x)

        return autograd.Variable(x), autograd.Variable(y)
    # end generate_training_data

    # Generate text data from text file
    def _generate_test_data(self, text):
        """
        Generate text data from text file
        :param text: Text
        :return: Test data set inputs
        """
        # Get inputs
        x = self._converter(text)

        # Convert to Tensor
        return autograd.Variable(torch.FloatTensor(x))
    # end generate_text_data

# end LSTMTextClassifier
