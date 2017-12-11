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
import Oger
import mdp
from datetime import datetime
from sys import getsizeof
from nsNLP.classifiers.TextClassifier import TextClassifier
import matplotlib.pyplot as plt
from decimal import *
import logging
import pickle
import os
from converters.PosConverter import PosConverter
from converters.TagConverter import TagConverter
from converters.WVConverter import WVConverter
from converters.FuncWordConverter import FuncWordConverter
from converters.OneHotConverter import OneHotConverter
from converters.LetterConverter import LetterConverter
import converters.JoinConverter


# ESN classifier model
class ESNTextClassifier(TextClassifier):
    """
    ESN classifier model
    """

    # Variables
    _verbose = False

    # Constructor
    def __init__(self, classes, size, leak_rate, input_scaling, w_sparsity, input_sparsity, spectral_radius, converter,
                 w=None, aggregation='average', use_sparse_matrix=False, smoothing=0.01, state_gram=1, parallel=False):
        """
        Constructor
        :param classes: Set of possible classes
        :param size: Reservoir's size
        :param leak_rate: Reservoir's leaky rate
        :param input_scaling: Input scaling
        :param w_sparsity: Hidden layer sparsity
        :param input_sparsity: Input layer sparsity
        :param spectral_radius: Hidden layer matrix's spectral radius
        :param converter: Word to input converter
        :param w: Hidden layer matrix
        :param aggregation: Aggregation function (average, multiplication)
        :param state_gram: Number of state to join
        """
        # Super
        super(ESNTextClassifier, self).__init__(classes=classes)

        # Properties
        self._input_dim = converter.get_n_inputs()
        self._output_dim = size
        self._leak_rate = leak_rate
        self._input_scaling = input_scaling
        self._w_sparsity = w_sparsity
        self._input_sparsity = input_sparsity
        self._spectral_radius = spectral_radius
        self._converter = converter
        self._last_y = []
        self._aggregation = aggregation
        self._author_token_count = np.zeros(self._n_classes)
        self._smoothing = smoothing
        self._state_gram = state_gram

        # Parallel
        if parallel and 'parallel' in mdp.get_extensions().keys():
            mdp.activate_extension('parallel')
        # end if

        # Create the reservoir
        self._reservoir = Oger.nodes.LeakyReservoirNode(input_dim=self._input_dim, output_dim=self._output_dim,
                                                        input_scaling=input_scaling,
                                                        leak_rate=leak_rate, spectral_radius=spectral_radius,
                                                        sparsity=input_sparsity, w_sparsity=w_sparsity, w=w,
                                                        use_sparse_matrix=use_sparse_matrix)

        # Components
        self._readout = None
        self._join = None
        self._last = None
        self._flow = None

        # Reset state at each call
        self._reservoir.reset_states = True

        # Init components
        self._reset_model()

        # Logger
        self._logger = logging.getLogger(name=u"RCNLP")
    # end __init__

    ##############################################
    # Properties
    ##############################################

    # Get last output
    @property
    def outputs(self):
        """
        Get last output
        :return:
        """
        return self._last_y
    # end outputs

    # Get w matrix
    def w(self):
        """
        Get w matrix
        :return:
        """
        return self._reservoir.w
    # end w

    ##############################################
    # Public
    ##############################################

    # Get w matrix
    def get_w(self):
        """
        Get w matrix
        :return:
        """
        return self._reservoir.w
    # end get_w

    # Get name
    def name(self):
        """
        Get name
        :return:
        """
        return u"ESN Text Classifier (size: {}, leaky-rate: {}, spectral-radius: {}, input-sparsity: {}, w-sparsity: {}".format(
            self._output_dim, self._leak_rate, self._spectral_radius, self._input_sparsity, self._w_sparsity)
    # end name

    # Show debugging informations
    def debug(self):
        """
        Show debugging informations
        """
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        plt.xlim([0, len(self._last_y[:, 0])])
        plt.ylim([0.0, 1.0])
        for author in range(self._n_classes):
            plt.plot(self._last_y[:, author], color=colors[author], label=u"Output {}".format(author))
            plt.plot(np.repeat(np.average(self._last_y[:, author]), len(self._last_y[:, author])), color=colors[author],
                     label=u"Output {} average".format(author), linestyle=u"dashed")
        # end for
        plt.show()
    # end debug

    # Get debugging data
    def get_debugging_data(self):
        """
        Get debugging data
        :return: debugging data
        """
        return self._last_y
    # end _get_debugging_data

    # Get embeddings
    def get_embeddings(self):
        """
        Get embeddings
        :return: Embedding matrix
        """
        # Embeddings
        embeddings = dict()

        # For each classes
        for c in self._classes:
            class_index = self._class_to_int(c)

            # Get
            embeddings[c] = self._readout.beta[:, class_index]
        # end for
        return embeddings
    # end get_embeddings

    ##############################################
    # Override
    ##############################################

    # To string
    def __str__(self):
        """
        To string
        :return:
        """
        return "ESNTextClassifier(n_classes={}, size={}, spectral_radius={}, leaky_rate={}, mem_size={}o)".format(
            self._n_classes, self._output_dim, self._spectral_radius, self._leak_rate, getsizeof(self))
    # end __str__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"ESNTextClassifier(n_classes={}, size={}, spectral_radius={}, leaky_rate={}, mem_size={}o)".format(
            self._n_classes, self._output_dim, self._spectral_radius, self._leak_rate, getsizeof(self))
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
        # Inputs outputs
        X = list()
        Y = list()

        # For each training text file
        for index, (x, y) in enumerate(self._examples):
            author_index = self._class_to_int(y)
            if verbose:
                print(u"Training on {}/{}...".format(index, len(self._examples)))
            # end if
            x, y = self._generate_training_data(x, author_index)
            if verbose:
                print(self._converter)
            # end if

            # Add inputs
            X.append(x)

            # Add outputs
            if self._state_gram == -1:
                Y.append(y[-1])
            else:
                Y.append(y)
            # end if

            # Token per classes
            self._author_token_count[author_index] += x.shape[0]
        # end for

        # Create data
        if self._state_gram == 1:
            data = [None, zip(X, Y)]
        else:
            data = [None, None, zip(X, Y)]
        # end if

        # Pre-log
        if verbose:
            print(u"Training model...")
            print(datetime.now().strftime("%H:%M:%S"))
        # end if

        # Train the model
        self._flow.train(data)

        # Post-log
        if verbose:
            print(datetime.now().strftime("%H:%M:%S"))
        # end if
    # end _finalize_training

    # Classify a text file
    def _classify(self, text):
        """
        Classify text
        :param text: Text to classify
        :return: Predicted class and class probabilities
        """
        # Get reservoir inputs
        x = self._generate_test_data(text)

        # Get reservoir response
        if self._state_gram == 1:
            y = self._flow(x)
        else:
            # Get states
            y = self._flow(x)
        # end if

        # Normalized
        y -= np.min(y)
        y /= np.max(y)

        # Save last y
        self._last_y = y

        # Get maximum probability class
        if self._aggregation == 'average':
            return self._int_to_class(np.argmax(np.average(y, 0))), np.average(y, 0)
        else:
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
        # end if
    # end _classify

    # Reset learning but keep reservoir
    def _reset_model(self):
        """
        Reset model learned parameters
        """
        # Delete old
        del self._readout, self._flow

        # Delete joiner if needed
        if self._state_gram > 1:
            del self._join
        # end if

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Flow
        if self._state_gram == -1:
            self._last = Oger.nodes.LastStateNode(input_dim=self._output_dim)
            self._flow = mdp.Flow([self._reservoir, self._last, self._readout], verbose=0)
        elif self._state_gram == 1:
            self._flow = mdp.Flow([self._reservoir, self._readout], verbose=0)
        else:
            self._join = Oger.nodes.JoinedStatesNode(input_dim=self._output_dim, joined_size=self._state_gram)
            self._flow = mdp.Flow([self._reservoir, self._join, self._readout], verbose=0)
        # end if

        # Examples
        self._examples = list()
    # end _reset_model

    # Generate training data from text
    def _generate_training_data(self, text, author):
        """
        Generate training data from text file.
        :param text: Text
        :param author: Corresponding author.
        :return: Data set inputs
        """
        # Get Temporal Representations
        reps = self._converter(text)

        # Converter type
        converter_type = type(self._converter)

        # Generate x and y
        return converter_type.generate_data_set_inputs(reps, self._n_classes, author)
    # end generate_training_data

    # Generate text data from text file
    def _generate_test_data(self, text):
        """
        Generate text data from text file
        :param text: Text
        :return: Test data set inputs
        """
        return self._converter(text)
    # end generate_text_data

    ##############################################
    # Static
    ##############################################

    # Generate W matrix
    @staticmethod
    def w(rc_size, rc_w_sparsity):
        """
        Generate W matrix
        :param rc_size:
        :param rc_w_sparsity:
        :return:
        """
        # W matrix
        w = mdp.numx.random.choice([0.0, 1.0], (rc_size, rc_size), p=[1.0 - rc_w_sparsity, rc_w_sparsity])
        w[w == 1] = mdp.numx.random.rand(len(w[w == 1]))
        return w
    # end w

    # Create ESN
    @staticmethod
    def create(classes, rc_size, rc_spectral_radius, rc_leak_rate, rc_input_scaling, rc_input_sparsity,
               rc_w_sparsity, converters_desc, w=None, voc_size=10000, uppercase=False,
               use_sparse_matrix=False, aggregation='average', pca_path="", state_gram=1, parallel=False, alphabet=u"",
               fill_in=False):
        """
        Constructor
        :param classes: Possible classes
        :param rc_size: Reservoir's size
        :param rc_spectral_radius: Reservoir's spectral radius
        :param rc_leak_rate: Reservoir's leak rate
        :param rc_input_scaling: Reservoir's input scaling
        :param rc_input_sparsity: Reservoir's input sparsity
        :param rc_w_sparsity: Reservoir's sparsity
        :param converters_desc: Input converter
        :param w:
        :param use_sparse_matrix:
        :param pca_path:
        :return:
        """
        # Converter list
        converter_list = list()

        # Joined converters
        joined_converters = True if len(converters_desc) > 1 else fill_in

        # For each converter
        for converter_desc in converters_desc:
            # Converter's info
            converter_type = converter_desc[0]
            converter_size = -1 if len(converter_desc) == 1 else converter_desc[1]

            # PCA model
            if converter_size != -1:
                pca_model = pickle.load(
                    open(os.path.join(pca_path, converter_type + unicode(converter_size) + u".p"), 'r'))
            else:
                pca_model = None
            # end if

            # Choose a text to symbol converter.
            if converter_type == "pos":
                converter = PosConverter(pca_model=pca_model, fill_in=joined_converters)
            elif converter_type == "tag":
                converter = TagConverter(pca_model=pca_model, fill_in=joined_converters)
            elif converter_type == "fw":
                converter = FuncWordConverter(pca_model=pca_model, fill_in=joined_converters)
            elif converter_type == "wv":
                converter = WVConverter(pca_model=pca_model, fill_in=joined_converters)
            elif converter_type == "oh":
                converter = OneHotConverter(voc_size=voc_size, uppercase=uppercase)
            elif converter_type == "ch":
                converter = LetterConverter(alphabet=alphabet)
            else:
                raise Exception(u"Unknown converter type {}".format(converter_desc))
            # end if

            # Add to list
            converter_list.append(converter)
        # end for

        # Join converters if necessary
        if len(converter_list) == 2:
            converter = converters.JoinConverter(converter_list[0], converter_list[1])
        else:
            converter = converter_list[0]
        # end if

        # Create the ESN Text Classifier
        classifier = ESNTextClassifier\
        (
            classes=classes,
            size=rc_size,
            input_scaling=rc_input_scaling,
            leak_rate=rc_leak_rate,
            input_sparsity=rc_input_sparsity,
            converter=converter,
            spectral_radius=rc_spectral_radius,
            w_sparsity=rc_w_sparsity,
            use_sparse_matrix=use_sparse_matrix,
            w=w,
            aggregation=aggregation,
            state_gram=state_gram,
            parallel=parallel
        )

        return classifier
    # end create_esn

# end ESNTextClassifier
