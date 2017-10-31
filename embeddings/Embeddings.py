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
from sklearn.metrics.pairwise import cosine_similarity


# Transform word to a vector
class Embeddings(object):
    """
    Transform word to a vector
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        :param lang: Language
        """
        # Globals
        self._word2vec = dict()
        self._properties = dict()
    # end __init__

    ############################################
    # Properties
    ############################################

    # Vocabulary size
    @property
    def voc_size(self):
        """
        Vocabulary size
        :return:
        """
        return len(self._word2vec.keys())
    # end voc_size

    ############################################
    # Public
    ############################################

    # Add a correspondence
    def add(self, word, vector):
        """
        Add a correspondance
        :param word:
        :param vector:
        :return:
        """
        self._word2vec[word] = vector
    # end add

    # Add multiple correspondences
    def adds(self, words, vectors):
        """
        Add multiple correspondences
        :param words:
        :param vectors:
        :return:
        """
        for index, word in enumerate(words):
            self._word2vec[word] = vectors[index]
        # end for
    # end adds

    # Set a property about a word
    def set(self, word, property, value):
        """
        Set a property about a word
        :param word:
        :param property:
        :param value:
        :return:
        """
        # Add dict
        if word not in self._properties:
            self._properties[word] = dict()
        # end if

        # Set properties
        self._properties[word][property] = value
    # end set

    # Remove a word from the correspondences
    def remove(self, word):
        """
        Remove a word from the correspondence
        :param word:
        :return:
        """
        del self._word2vec[word]
    # end remove

    # Clean the correspondences based on a property
    def clean(self, property, value, threshold='min'):
        """
        Clean the correspondences based on a property
        :param property:
        :param value:
        :param threshold:
        :return:
        """
        for word in self._properties.keys():
            if property in self._properties[word].keys():
                if threshold == 'min':
                    # MIN
                    if self._properties[word][property] < value:
                        self.remove(word)
                    # end if
                elif threshold == 'max':
                    # MAX
                    if self._properties[word][property] > value:
                        self.remove(word)
                    # end if
                elif threshold == 'eq':
                    # EQUAL
                    if self._properties[word][property] == value:
                        self.remove(word)
                    # end if
                elif threshold == 'neq':
                    # NOT EQUAL
                    if self._properties[word][property] != value:
                        self.remove(word)
                    # end if
                else:
                    raise Exception(u"Unknown threshold")
                # end if
            # end if
        # end for
    # end clean

    # Similar words
    def similar_words(self, vector, count=20, measure_func='cosine'):
        """
        Similar words
        :param vector:
        :return:
        """
        # List of words with measures
        word_list = list()

        # For each word in dictionary
        for word in self._word2vec:
            # Vector
            word_vector = self[word]

            # Measure
            if measure_func == 'cosine':
                reverse = True
                measure = self._cosine_similarity(vector, word_vector)
            elif measure_func == 'euclidian':
                reverse = False
                measure = self._euclidian_distance(vector, word_vector)
            else:
                raise Exception(u"Unknown measure")
            # end if

            # Add to list
            word_list.append((word, measure))
        # end for

        # Sort the list by measure
        word_list.sort(key=lambda tup: tup[1], reverse=reverse)

        # Return
        return word_list[:count]
    # end similar_words

    ############################################
    # Private
    ############################################

    # Euclidian distance
    def _euclidian_distance(self, vec1, vec2):
        """
        Euclidian distance
        :param vec1:
        :param vec2:
        :return:
        """
        return np.linalg.norm(vec1-vec2)
    # end euclidian_distance

    # Cosine similarity
    def _cosine_similarity(self, vec1, vec2):
        """
        Cosine similarity
        :param vec2:
        :return:
        """
        return cosine_similarity(vec1, vec2)
    # end

    ############################################
    # Override
    ############################################

    # Get vectors from word-symbols
    def __call__(self, words):
        """
        Get vectors from word-symbols
        :param items:
        :return:
        """
        # List of vectors
        vectors = list()

        # For each word-symbol
        for word in words:
            vectors.append(self[word])
        # end for

        return vectors
    # end __call__

    # Get a vector from a word-symbol
    def __getitem__(self, word):
        """
        Override get item
        :param item:
        :return:
        """
        if word in self._word2vec:
            return self._word2vec[word]
        else:
            return None
        # end if
    # end __getitem__

# end Embeddings
