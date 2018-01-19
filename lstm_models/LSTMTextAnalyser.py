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
from LSTMTextClassifier import LSTMTextClassifier
from sys import getsizeof
import torch


# LSTM analyser model
class LSTMTextAnalyser(LSTMTextClassifier):
    """
    LSTM analyser model
    """

    # Constructor
    def __init__(self, classes, hidden_size, converter, embedding_dim=300, learning_rate=0.1, voc_size=0, n_epoch=300,
                 aggregation='average', smoothing=0.001):
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
        super(LSTMTextAnalyser, self).__init__(classes, hidden_size, converter, embedding_dim, learning_rate, voc_size,
                                               n_epoch, aggregation, smoothing)
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
        return "LSTMTextAnalyser(n_classes={}, embedding_size={}, hidden_size={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, getsizeof(self))

    # end __str__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"LSTMTextAnalyser(n_classes={}, embedding_size={}, hidden_size={}, mem_size={}o)".format(
            self._n_classes, self._embedding_dim, self._hidden_size, getsizeof(self))
    # end __unicode__

    ##############################################
    # Private
    ##############################################

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
