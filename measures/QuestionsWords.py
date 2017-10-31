#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the nsNLP Project.
# The nsNLP Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# nsNLP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
#

import os
import codecs
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
import numpy as np
import dataset.questions_words


# Questions words measures
class QuestionsWords(object):
    """
    Questions words
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Properties
        self._questions_words = list()

        # Load dataset
        self._load()
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Test words embeddings with linear positioning
    def linear_positioning(self, word_embeddings, measure='cosine'):
        """
        Test words embeddings with linear positioning
        :param word_embeddings:
        :return:
        """
        # Linear positions
        linear_positions = list()

        # For each questions-words
        for (word1, word2, word3, response_word) in self._questions_words:
            # Get all vectors
            vec1 = word_embeddings[word1]
            vec2 = word_embeddings[word2]
            vec3 = word_embeddings[word3]
            observation = word_embeddings[response_word]

            # No unknown words
            if vec1 is not None and vec2 is not None and vec3 is not None and observation is not None:
                # Get nearest word
                predictions = word_embeddings.similar_words(vec1 - vec2 + vec3, count=word_embeddings.voc_size(),
                                                            measure_func=measure)

                # Find the position
                for index, (pred_word, pred_measure) in enumerate(predictions):
                    if pred_word == response_word:
                        linear_pos = 1.0 - (float(index) / float(word_embeddings.voc_size()))
                        linear_positions.append(linear_pos)
                        break
                    # end if
                # end for
            # end if
        # end for

        return np.average(linear_positions), linear_positions
    # end linear_positioning

    # Test several embeddings and compare with t-test
    def embeddings_significance(self, embeddings_list, test='linear_positioning', measure='cosine'):
        """
        Test several embeddings and compare with t-test
        :param embeddings_list:
        :param test:
        :param measure:
        :return:
        """
        # Significance matrix
        significance_matrix = np.zeros((len(embeddings_list), len(embeddings_list)))

        # List of positioning
        positionings = np.zeros(len(embeddings_list))

        # For each couple of embeddings
        for index1, emb1 in enumerate(embeddings_list):
            for index2, emb2 in enumerate(embeddings_list):
                if emb1 != emb2 and emb1.voc_size() == emb2.voc_size():
                    # Test embeddings
                    if test == 'linear_positioning':
                        average_pos1, positions1 = self.linear_positioning(emb1, measure=measure)
                        average_pos2, positions2 = self.linear_positioning(emb2, measure=measure)
                    # end if

                    # Save positioning
                    positionings[index1] = average_pos1
                    positionings[index2] = average_pos2

                    # T-test on two related samples
                    t_test_p_value = scipy.stats.ttest_rel(positions1, positions2).pvalue
                    significance_matrix[index1, index2] = t_test_p_value
                # end if
            # end for
        # end for

        return positionings, significance_matrix
    # end embeddings_significance

    ###########################################
    # Private
    ###########################################

    # Load the dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # Questions string
        questions_string = dataset.questions_words.questions_words

        # Split by line
        question_lines = questions_string.split(u'\n')

        # For each line
        for line in question_lines:
            # No comments
            if line[0] != u':':
                # Split for each words
                question_words = line.split(u' ')

                # Add to examples
                self._questions_words.append(question_words)
            # end if
        # end for
    # end _load

# end QuestionsWords
