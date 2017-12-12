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
from Converter import Converter
import scipy.sparse


# Converter from letter to symbols
class LetterConverter(Converter):
    """
    Convert letter to symbols
    """

    # Constructor
    def __init__(self, alphabet, tag_to_symbol=None, resize=-1, pca_model=None, fill_in=False):
        """
        Constructor
        :param alphabet:
        :param unknown:
        :param lang: Language model
        :param tag_to_symbol: Tag to symbol conversion array.
        :param resize: Reduce dimensionality.
        """
        # Properties
        self._alphabet = alphabet
        self._fill_in = fill_in
        self._n_chars = len(alphabet) + 1

        # Super
        super(LetterConverter, self).__init__(tag_to_symbol, resize, pca_model)

        # Letter to index
        self._char2index = dict()
        for index, c in enumerate(alphabet):
            self._char2index[c] = index
        # end for
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A tag list.
        """
        return self._alphabet
        #return [u' ', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p',
        #        u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'.', u',', u';', u'-', u'!', u'?']
    # end get_tags

    ##############################################
    # Override
    ##############################################

    # Convert a string to a ESN input
    def __call__(self, text, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param text: The text to convert.
        :return: An numpy array of inputs.
        """
        # Resulting sparse matrix
        doc_array = scipy.sparse.csr_matrix((len(text), self._n_chars))

        # For each letter
        for pos, letter in enumerate(text):
            # Try to get char
            try:
                index = self._char2index[letter]
            except KeyError:
                index = -1
            # end try

            # Set input
            doc_array[pos, index] = 1.0
        # end for

        return doc_array
    # end convert

    ##############################################
    # Private
    ##############################################

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return: The input size.
        """
        return self._n_chars
    # end if

# end LetterConverter
