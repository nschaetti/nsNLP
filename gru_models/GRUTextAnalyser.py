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
from nsNLP.classifiers.TextClassifier import TextClassifier
from GRUTextClassifier import GRUTextClassifier
from sys import getsizeof
import torch


# LSTM analyser model
class GRUTextAnalyser(GRUTextClassifier):
    """
    GRU analyser model
    """

    # Constructor
    def __init__(self, classes, hidden_size, converter, n_layers=1, embedding_dim=300, learning_rate=0.1, voc_size=0,
                 n_epoch=300, aggregation='average', smoothing=0.001, word2index=None, tokenizer=None):
        """
        Constructor
        :param classes:
        :param hidden_size:
        :param converter:
        :param embedding_dim:
        :param learning_rate:
        :param voc_size:
        :param n_epoch:
        :param aggregation:
        :param smoothing:
        """
        super(GRUTextAnalyser, self).__init__(classes, hidden_size, converter, n_layers, embedding_dim, learning_rate, voc_size,
                                               n_epoch, aggregation, smoothing, word2index, tokenizer)
    # end __init__

    ##############################################
    # Properties
    ##############################################

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Override
    ##############################################

    # To string
    def __str__(self):
        """
        To string
        :return:
        """
        return "GRUTextAnalyser(n_classes={}, embedding_size={}, hidden_size={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, getsizeof(self))

    # end __str__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"GRUTextAnalyser(n_classes={}, embedding_size={}, hidden_size={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, getsizeof(self))
    # end __unicode__

    ##############################################
    # Private
    ##############################################

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
                train_x, _ = self._generate_training_data(x, class_index)

                # For each target time t
                train_y = [self._int_to_class(tt) for tt in y]
                train_y = torch.LongTensor(train_y)

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
        _, best_classes = torch.max(y, 1)

        # Classes
        return [self._int_to_class(x) for x in best_classes]
    # end _classify

# end LSTMTextAnalyser
