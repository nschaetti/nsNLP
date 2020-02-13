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
from .dataset import questions_words
import unicodecsv as csv
import math
from multiprocessing import Queue, Process, Lock


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
    # Properties
    ###########################################

    # Number of queries
    @property
    def size(self):
        """
        Number of queries
        :return:
        """
        return len(self._questions_words)
    # end size

    ###########################################
    # Public
    ###########################################

    # Test words embeddings with positioning
    def positioning(self, word_embeddings, csv_file, measure='cosine', func='linear', n_threads=1):
        """
        Test words embeddings with linear positioning
        :param word_embeddings:
        :return:
        """
        # Linear positions
        positionings = list()

        # Total
        total = float(len(self._questions_words))

        # Write CSV
        with open(csv_file, 'wb') as f:
            # CSV writer
            csv_writer = csv.writer(f, )

            # Write header
            head_row = [u"word1", u"word2", u"word3", u"result", u"position", u"measure"] + [str(i+1) for i in range(100)]

            # Queue
            q = Queue()

            # Locker
            lock = Lock()

            # Write header
            self._write_positioning_csv(lock, csv_writer, head_row)

            # Step
            step = math.ceil(total / n_threads)

            # Launch each process
            for start in np.arange(0, total, step):
                Process(target=self._evalute_positioning, args=(q, lock, csv_writer, word_embeddings, int(start), int(start+step), measure, func)).start()
            # end for

            # Wait for threads

            # Get results
            for num in range(n_threads):
                positionings += q.get(block=True)
            # end for
        # end with

        # Compute positioning
        if func == 'linear':
            mu = 0.0
        elif func == 'inv':
            mu = math.log(word_embeddings.voc_size) / float(word_embeddings.voc_size)
        else:
            raise Exception(u"Unknown positioning function")
        # end if

        return np.average(positionings), positionings, float(len(positionings)) / float(
            len(self._questions_words)), scipy.stats.ttest_1samp(positionings, popmean=mu).pvalue
    # end linear_positioning

    # Test several embeddings and compare with t-test
    def embeddings_significance(self, embeddings_list, measure='cosine', func='linear'):
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
                    average_pos1, positions1 = self.positioning(emb1, measure=measure, func=func)
                    average_pos2, positions2 = self.positioning(emb2, measure=measure, func=func)

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

    # Write in positioning file
    def _write_positioning_csv(self, lock, csv_writer, row):
        """
        Write in positioning CSV file
        :param csv_file:
        :param row:
        :return:
        """
        # Lock
        lock.acquire()

        # Write to CSV
        csv_writer.writerow(row)

        # Release
        lock.release()
    # end _write_positioning_csv

    # Evaluate positioning
    def _evalute_positioning(self, queue, lock, csv_writer, word_embeddings, start, end, measure='cosine', func='linear'):
        """
        Evaluate positioning
        :param word_embeddings:
        :param csv_file:
        :param measure:
        :param func:
        :return:
        """
        # Linear positions
        positionings = list()

        # For each questions-words
        for (word1, word2, word3, response_word) in self._questions_words[start:end]:
            # Get all vectors
            vec1 = word_embeddings[word1]
            vec2 = word_embeddings[word2]
            vec3 = word_embeddings[word3]
            observation = word_embeddings[response_word]

            # No unknown words
            if vec1 is not None and vec2 is not None and vec3 is not None and observation is not None:
                # Get nearest word
                predictions = word_embeddings.similar_words(vec1 - vec2 + vec3, count=word_embeddings.voc_size,
                                                            measure_func=measure)

                # Find the position
                for index, (pred_word, pred_measure) in enumerate(predictions):
                    if pred_word == response_word:
                        word_pos = index + 1
                        break
                        # end if
                # end for

                # Compute positioning
                if func == 'linear':
                    word_positioning = self._linear_positioning(word_pos, word_embeddings.voc_size)
                elif func == 'inv':
                    word_positioning = self._inverse_positioning(word_pos, word_embeddings.voc_size)
                else:
                    raise Exception(u"Unknown positioning function")
                # end if

                # Add to total positionings
                positionings.append(word_positioning)

                # Word row
                word1_row = u"{} ({})".format(word1, word_embeddings.get(word1, 'count'))
                word2_row = u"{} ({})".format(word2, word_embeddings.get(word2, 'count'))
                word3_row = u"{} ({})".format(word3, word_embeddings.get(word3, 'count'))
                response_word_row = u"{} ({})".format(response_word, word_embeddings.get(response_word, 'count'))

                # Row array
                row = [word1_row, word2_row, word3_row, response_word_row, word_pos, word_positioning] + [str(el) for el in predictions[:40]]

                # Write to CSV
                self._write_positioning_csv(lock, csv_writer, row)
            # end if
        # end for

        # Return through queue
        queue.put(positionings)
    # end _evalute_positioning

    # Linear positioning
    def _linear_positioning(self, word_pos, voc_size):
        """
        Linear positioning
        :param word_pos:
        :param voc_size:
        :return:
        """
        return -((2.0*float(word_pos))/float(voc_size)) + 1.0
    # end _linear_positioning

    # Inverse positioning
    def _inverse_positioning(self, word_pos, voc_size):
        """
        Inverse positioning
        :param word_pos:
        :param voc_size:
        :return:
        """
        return 1.0 / word_pos
    # end _inverse_positioning

    # Load the dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # Questions string
        questions_string = questions_words.questions_words

        # Split by line
        question_lines = questions_string.split(u'\n')

        # For each line
        for line in question_lines:
            if len(line) > 0:
                # No comments
                if line[0] != u':':
                    # Split for each words
                    question_words = line.split(u' ')

                    # Add to examples
                    self._questions_words.append(question_words)
                # end if
            # end if
        # end for
    # end _load

# end QuestionsWords
