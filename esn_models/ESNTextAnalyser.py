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
from datetime import datetime
from sys import getsizeof
from nsNLP.esn_models.ESNTextClassifier import ESNTextClassifier
import pickle
import os
from converters.PosConverter import PosConverter
from converters.TagConverter import TagConverter
from converters.WVConverter import WVConverter
from converters.FuncWordConverter import FuncWordConverter
from converters.OneHotConverter import OneHotConverter
from converters.LetterConverter import LetterConverter
import converters.JoinConverter


# ESN analyser model
class ESNTextAnalyser(ESNTextClassifier):
    """
    ESN analyser model
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
        super(ESNTextAnalyser, self).__init__(classes=classes, size=size, leak_rate=leak_rate,
                                              input_scaling=input_scaling, w_sparsity=w_sparsity,
                                              input_sparsity=input_sparsity, spectral_radius=spectral_radius,
                                              converter=converter, aggregation=aggregation,
                                              use_sparse_matrix=use_sparse_matrix, smoothing=smoothing,
                                              state_gram=state_gram, parallel=parallel)
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
        return "ESNTextAnalyser(n_classes={}, size={}, spectral_radius={}, leaky_rate={}, mem_size={}o)".format(
            self._n_classes, self._output_dim, self._spectral_radius, self._leak_rate, getsizeof(self))
    # end __str__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"ESNTextAnalyser(n_classes={}, size={}, spectral_radius={}, leaky_rate={}, mem_size={}o)".format(
            self._n_classes, self._output_dim, self._spectral_radius, self._leak_rate, getsizeof(self))
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
        # Inputs outputs
        X = list()
        Y = list()

        # For each training text file
        for index, (x, y) in enumerate(self._examples):
            if verbose:
                print(u"Training on {}/{}...".format(index, len(self._examples)))
            # end if
            x, _ = self._generate_training_data(x, 0)

            if verbose:
                print(self._converter)
            # end if

            # Length
            try:
                length_x = len(x)
                length_y = len(y)
            except TypeError:
                length_x = x.shape[0]
                length_y = y.shape[0]
            # end try

            # Check not empty
            if length_x > 0 and length_y > 0:
                if length_x == length_y:
                    # Add inputs
                    X.append(x)

                    # Add outputs
                    if self._state_gram == -1:
                        Y.append(y[-1])
                    else:
                        Y.append(y)
                    # end if
                else:
                    print(u"Error input and output are not the same length ({}/{})!".format(len(x), len(y)))
                    exit()
                # end if
            else:
                print(u"Warning input or output is empty, removed ({}/{})!".format(len(x), len(y)))
            # end if
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

        # Get maximum probability class at each time t
        return np.argmax(y, axis=1)
    # end _classify

    ##############################################
    # Static
    ##############################################

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
        classifier = ESNTextAnalyser\
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
