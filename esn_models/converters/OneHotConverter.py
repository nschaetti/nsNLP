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

import numpy as np
import pickle
import re
from Converter import Converter
import scipy.sparse as sp

###########################################################
# Exceptions
###########################################################


# One-hot vector representations are full
class OneHotVectorFullException(Exception):
    """
    One-hot vector representations are full
    """
    pass
# end OneHotVectorFullException

###########################################################
# Class
###########################################################


# Convert text to one-hot vectors
class OneHotConverter(Converter):
    """
    Convert text to one-hot vectors
    """

    # Constructor
    def __init__(self, voc_size=5000, uppercase=False):
        """
        Constructor
        :param dim: Input vector dimension
        :param voc_size: Vocabulary size
        :param uppercase: Use uppercase
        """
        # Super class
        super(OneHotConverter, self).__init__(None, -1, None)

        # Properties
        self._voc_size = voc_size
        self._voc = dict()
        self._word_pos = 0
        self._word_index = dict()
        self._index_word = dict()
        self._word_counter = dict()
        self._total_counter = 0
        self._uppercase = uppercase
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Words
    def words(self):
        """
        Words
        :return:
        """
        return self._voc.keys()
    # end words

    # Vocabulary
    def voc(self):
        """
        Vocabulary
        :return:
        """
        return self._voc
    # end voc

    # Get matrix
    def get_matrix(self):
        """
        Get matrix
        :return:
        """
        words_matrix = np.zeros((len(self._voc.keys()), self._voc_size))
        for index, word in enumerate(self.words()):
            words_matrix[index, :] = self._voc[word]
        # end for
        return words_matrix
    # end get_matrix

    # Get word by index
    def get_word_by_index(self, index):
        """
        Get word by index
        :param index: Index
        :return:
        """
        if index < len(self._index_word):
            return self._index_word[index]
        else:
            return None
        # end if
    # end get_word_by_index

    # Get word count
    def get_n_words(self):
        """
        Get word count
        :return: word count
        """
        return self._word_pos
    # end get_n_words

    # Get word count
    def get_word_count(self, word):
        """
        Get word count
        :param word:
        :return:
        """
        try:
            return self._word_counter[word.lower()]
        except KeyError:
            return 0
        # end try
    # end get_word_count

    # Get word counts
    def get_word_counts(self):
        """
        Get word counts
        :return:
        """
        return self._word_counter
    # end get_word_counts

    # Reset word count
    def reset_word_count(self):
        """
        Reset word count
        :return:
        """
        self._word_counter = dict()
        self._total_counter = 0
    # end if

    # Get total word count
    def get_total_count(self):
        """
        Get total word count
        :return:
        """
        return self._total_counter
    # end get_total_count

    # Set word indexes
    def set_word_indexes(self, word_indexes):
        """
        Set word indexes
        :param word_indexes:
        :return:
        """
        self._word_index = word_indexes

    # Get word index
    def get_word_index(self, word_text):
        """
        Get word index
        :param word_text:
        :return:
        """
        return self._word_index[word_text]

    # end get_word_index

    # Get word indexes
    def get_word_indexes(self):
        """
        Get word indexes
        :return:
        """
        return self._word_index
    # end get_word_indexes

    # Create a new word vector randomly
    def create_word_vector(self, word):
        """
        Create a new word vector randomly
        :param word: The word
        """
        if self._word_pos < self._voc_size:
            self._voc[word] = self._one_hot()
            self._word_index[word] = self._word_pos - 1
            self._index_word[self._word_pos - 1] = word
        else:
            raise OneHotVectorFullException("One-hot vector representations are full")
        # end if
    # end create_word_vector

    # Save Word2Vec
    def save(self, file_name):
        """
        Save Word2Vec
        :param file_name:
        :return:
        """
        pickle.dump(self, open(file_name, 'wb'))
    # end save

    ##############################################
    # Override
    ##############################################

    # Get a word vector
    def __getitem__(self, item):
        """
        Get a word vector.
        :param item: Item to retrieve, if does not exists, create it.
        :return: The attribute value
        """
        # Transform
        item = self._transform_word(item)

        # Create word if needed
        if item not in self._voc.keys():
            self.create_word_vector(item)
        # end if

        return self._voc[item]
    # end __getattr__

    # Set a word vector
    def __setitem__(self, word, vector):
        """
        Set a word vector.
        :param word: Word to set
        :param vector: New word's vector
        """
        # Transform
        word = self._transform_word(word)

        # Word to vector
        self._voc[word] = vector
    # end if

    # Convert a string to a ESN input
    def __call__(self, tokens, exclude=list(), word_exclude=list()):
        """
        Convert a string to a ESN input
        :param tokens: The text to convert.
        :return: An numpy array of inputs.
        """
        # Resulting numpy array
        doc_array = np.array([])

        # For each token
        ok = False
        for index, word in enumerate(tokens):
            if word not in exclude:
                # Transform text
                word_text = self._transform_word(word)
                word_text = word_text.replace(u"\n", u"")
                word_text = word_text.replace(u"\t", u"")
                word_text = word_text.replace(u"\r", u"")

                # Replacement
                word_text = OneHotConverter.replace_token(word_text, r"^[0-9]{4}\-[0-9]{4}$", u"<interval>")
                word_text = OneHotConverter.replace_token(word_text, r"^[0-9]{4}\-[0-9]{2}$", u"<interval>")
                word_text = OneHotConverter.replace_token(word_text, r"^\d+th$", u"<th>")
                word_text = OneHotConverter.replace_token(word_text, r"^\d+nd$", u"<th>")
                word_text = OneHotConverter.replace_token(word_text, r"^[+-]?\d+(?:\.\d+)?\%$", u"<percent>")
                word_text = OneHotConverter.replace_token(word_text, r"^[+-]?\d+(?:\.\d+)+$", u"<float>")
                word_text = OneHotConverter.replace_token(word_text, r'^\d+(?:,\d+)+$', u"<number>")
                word_text = OneHotConverter.replace_token(word_text, r"^[0-9]{4}$", u"<4digits>")
                word_text = OneHotConverter.replace_token(word_text, r"^[0-9]{3}$", u"<3digits>")
                word_text = OneHotConverter.replace_token(word_text, r"^[0-9]{2}$", u"<2digits>")
                word_text = OneHotConverter.replace_token(word_text, r"^[0-9]{1}$", u"<1digit>")
                word_text = OneHotConverter.replace_token(word_text, r"^[+-]?\d+$", u"<integer>")

                # No empty word
                if len(word_text) > 0:
                    if not ok:
                        doc_array = self[word_text]
                        ok = True
                    else:
                        doc_array = sp.vstack((doc_array, self[word_text]))
                    # end if
                # end if
            # end if
        # end for

        return self.reduce(doc_array)
    # end convert

    # Left multiplication
    def __mul__(self, other):
        """
        Left multiplication
        :param other:
        :return:
        """
        for word in self._voc.keys():
            self._voc[word] *= other
        # end for
        return self

    # end __mul__

    # Right multiplication
    def __rmul__(self, other):
        """
        Right multiplication
        :param other:
        :return:
        """
        for word in self._voc.keys():
            self._voc[word] *= other
        # end for
        return self
    # end __rmul__

    # Augmented assignment (mult)
    def __imul__(self, other):
        """
        Augmented assignment (mult)
        :param other:
        :return:
        """
        for word in self._voc.keys():
            self._voc[word] *= other
        # end for
        return self
    # end __imul__

    ##############################################
    # Private
    ##############################################

    # Get the number of inputs
    def _get_inputs_size(self):
        """
        Get the input size.
        :return: The input size.
        """
        return self._voc_size
    # end get_n_inputs

    # Map word to a one-hot vector
    def _one_hot(self):
        """
        Map word to a one-hot vector
        :return: A new one-hot vector
        """
        vec = np.zeros(self._voc_size, dtype='float64')
        vec[self._word_pos] = 1.0
        vec = sp.csr_matrix(vec)
        self._word_pos += 1
        return vec
    # end one_hot

    # Transform word
    def _transform_word(self, word_text):
        """
        Transform word
        :param word_text:
        :return:
        """
        if not self._uppercase:
            return word_text.lower()
        # end if
        return word_text
    # end _transform_word

    ##############################################
    # Static
    ##############################################

    # Replace token if match a regex
    @staticmethod
    def replace_token(token, regex, repl):
        """
        Replace token if match a regex
        :param token:
        :param regex:
        :param repl:
        :return:
        """
        if re.match(regex, token):
            return repl
        # end if
        return token
    # end replace_token

    # Generate data set inputs
    @staticmethod
    def generate_data_set_inputs(reps, n_outputs, output_pos):
        """
        Generate data set inputs
        :param reps:
        :param n_outputs:
        :param output_pos:
        :return:
        """
        # Number of representations
        n_reps = reps.shape[0]

        # Author vector
        outputs = sp.csr_matrix((n_reps, n_outputs))
        outputs[:, output_pos] = 1.0

        return reps, outputs
    # end generate_data_set_inputs

# end RCNLPConverter
