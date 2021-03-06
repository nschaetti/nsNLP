#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : classifiers.TextClassifier.py
# Description : Text classifier abstract class.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
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

# Import
import pickle
import numpy as np


# Text generator
class TextGenerator(object):
    """
    Text generator
    """

    # Text classifier
    _training_finalized = False

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes: Classes
        :param lang: Spacy language
        """
        # Properties
        self._classes = classes
        self._n_classes = len(classes)
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get name
    def name(self):
        """
        Get name
        :return:
        """
        pass
    # end name

    # Train the model
    def train(self, x, y):
        """
        Train the model
        :param x: Example's inputs.
        :param y: Example's outputs.
        """
        if not self._training_finalized:
            return self._train(x, y)
        else:
            return False
        # end if
    # end train

    # Train a set
    def training(self, samples):
        """
        Train a set
        :param samples:
        :param verbose:
        :return:
        """
        for sample in samples:
            self.train(sample[0], sample[1])
        # end for
    # end training

    # Finalize model training
    def finalize(self, verbose=False):
        """
        Finalize model training
        """
        if not self._training_finalized:
            self._finalize_training(verbose)
            self._training_finalized = True
        # end if
    # end finalize

    # Generate text
    def generate(self, x, text_length):
        """
        Generate text from learning
        :param x: Sample
        :param text_length: Generated text's length
        :return: Predicted class and classes probabilities
        """
        return self._generate(x, text_length)
    # end predict

    # Reset the classifier
    def reset(self):
        """
        Reset the classifier
        """
        self._reset_model()
        self._training_finalized = False
    # end reset

    # Show the debuging informations
    def debug(self):
        """
        Show the debugging informations
        :return:
        """
        pass
    # end debug

    # Get debugging data
    def get_debugging_data(self):
        """
        Get debugging data
        :return: debugging data
        """
        pass
    # end _get_debugging_data

    # Save the model
    def save(self, filename):
        """
        Save the model to a Pickle file
        :param filename:
        :return:
        """
        with open(filename, 'w') as f:
            pickle.dump(self, f)
        # end with
    # end save

    # Finalized?
    def finalized(self):
        """
        Finalized?
        :return:
        """
        return self._training_finalized
    # end finalized

    ##############################################
    # Override
    ##############################################

    # Generate text from a starting point
    def __call__(self, x, text_length):
        """
        Generate text from a starting point
        :param x: Starting point as a serie of tokens
        :param text_length: Generated text's length
        :return: A serie of generated tokens
        """
        # Finalize training
        if not self._training_finalized:
            self._finalize_training()
            self._training_finalized = True
        # end if

        # Classify the document
        return self._generate(x, text_length)
    # end __class__

    # To str
    def __str__(self):
        """
        To string
        :return:
        """
        pass
    # end __str__

    ##############################################
    # Private
    ##############################################

    # Train the model
    def _train(self, x, c, verbose=False):
        """
        Train
        :param x: Example's inputs
        :param c: Example's outputs
        :param verbose: Verbosity
        """
        pass
    # end _train

    # Filter token
    def _filter_token(self, word):
        """
        Filter token
        :param token:
        :return:
        """
        word_text = word.text
        word_text = word_text.replace(u"\n", u"")
        word_text = word_text.replace(u"\t", u"")
        word_text = word_text.replace(u"\r", u"")
        if len(word_text) > 0:
            word_vector = word.vector
            if np.average(word_vector) != 0:
                return True, word_text
            # end if
        # end if
        return False, ""
    # end if

    # Generate a series of tokens from a starting point
    def _generate(self, x, text_length):
        """
        Generate a series of tokens from a starting point
        :param x: A starting point as a series of tokens
        :param text_length:
        :return: A series of tokens
        """
        pass
    # end _classify

    # Finalize the training
    def _finalize_training(self, verbose=False):
        """
        Finalize training.
        :param verbose: Verbosity
        """
        pass
    # end _finalize_training

    # Reset the model
    def _reset_model(self):
        """
        Reset the model
        """
        pass
    # end _reset_model

    # Transform int to class name
    def _int_to_class(self, index):
        """
        Transform index to class name.
        :param class_index: Class index.
        :return: Class name.
        """
        return self._classes[index]
    # end _int_to_class

    # Transform class name to int
    def _class_to_int(self, class_name):
        """
        Transform class name to int
        :param class_name: Class name
        :return: Integer
        """
        for index, name in enumerate(self._classes):
            if name == class_name:
                return index
            # end if
        # end for
        return -1
    # end class_to_int

    ##########################################
    # Static
    ##########################################

    # Load the model
    @staticmethod
    def load(opt):
        """
        Load the model from a file
        :param opt: Loading option
        :return: The model object
        """
        return pickle.load(open(opt, 'r'))
    # end load

# end TextClassifier
