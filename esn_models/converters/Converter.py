#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
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

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Base class for converters
class Converter(object):
    """
    Base class for converters.
    """

    # Constructor
    def __init__(self, tag_to_symbol=None, resize=-1, pca_model=None, upper_level=None):
        """
        Constructor
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        # Properties
        self._resize = resize
        self._pca_model = pca_model
        self._upper_level = upper_level

        # Generate tag symbols
        if tag_to_symbol is None:
            self._symbols = self.generate_symbols()
        else:
            self._symbols = tag_to_symbol
            # end if
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Generate symbols
    def generate_symbols(self):
        """
        Generate word symbols.
        :return: Dictionary of tag to symbols.
        """
        result = dict()
        n_words = len(self.get_tags())
        for index, p in enumerate(self.get_tags()):
            result[p] = np.zeros(n_words)
            result[p][index] = 1.0
        # end for
        return result
    # end generate_symbols

    # Get symbol from tag
    def tag_to_symbol(self, tag):
        """
        Get symbol from tag.
        :param tag: Tag.
        :return: The corresponding symbols.
        """
        if tag in self._symbols.keys():
            return self._symbols[tag]
        return None
    # end word_to_symbol

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return:
        """
        return []

    # end get_tags

    # Reduce the inputs
    def reduce(self, x):
        """
        Reduce the inputs.
        :param x: The signal to reduce.
        :return: The reduce signal.
        """
        if self._pca_model is not None:
            return self._pca_model.transform(x)
        elif self._resize != -1:
            # PCA
            pca = PCA(n_components=self._resize)
            pca.fit(x)
            return pca.transform(x)
        # end if
        return x
    # end reduce

    # Get the number of inputs
    def get_n_inputs(self):
        """
        Get the number of inputs.
        :return: The input size.
        """
        if self._pca_model is not None:
            return self._pca_model.n_components_
        elif self._resize != -1:
            return self._resize
        else:
            return self._get_inputs_size()
        # end if
    # end get_n_inputs

    ##############################################
    # Override
    ##############################################

    # Convert a string to a ESN input
    def __call__(self, tokens, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param tokens: Text to convert.
        :param exclude: List of tags to exclude.
        :param word_exclude: List of words to exclude.
        :return: A list of symbols.
        """
        pass
    # end convert

    ##############################################
    # Private
    ##############################################

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return:
        """
        return 0
    # end if

    ##############################################
    # Static
    ##############################################

    # Display representations
    @staticmethod
    def display_representations(rep):
        """
        Display representations
        :param rep:
        :return:
        """
        plt.imshow(rep, cmap='Greys')
        plt.show()
    # end display_representations

    # Generate data set inputs
    @staticmethod
    def generate_data_set_inputs(reps, n_authors, author):
        """
        Generate data set inputs
        :param reps:
        :param n_authors:
        :param author:
        :return:
        """
        # Number of representations
        n_reps = reps.shape[0]

        # Author vector
        author_vector = np.zeros((1, n_authors))
        author_vector[0, author] = 1.0

        # Output
        outputs = np.repeat(author_vector, n_reps, axis=0)

        return reps, outputs
    # end generate_data_set_inputs

# end Converter
