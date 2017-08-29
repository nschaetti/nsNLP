# -*- coding: utf-8 -*-
#
# File : core/downloader/PySpeechesConfig.py
# Description : .
# Date : 20th of February 2017
#
# This file is part of pySpeeches.  pySpeeches is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Import packages
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import torch
import sys


# A matrix with 2-gram of letters frequencies
class LettersMatrix(object):
    """
    A matrix with 2-gram of letters frequencies
    """

    # Constructor
    def __init__(self, features_mapping, letters="", punctuations=""):
        """
        Constructor
        :param letters:
        :param punctuations:
        :param upper_case:
        """
        # Properties
        self._features_mapping = features_mapping
        self._letters = letters
        self._grams_col = 0
        self._end_grams_col = len(letters)
        self._punctuations_col = self._end_grams_col + 1
        self._first_grams_col = self._punctuations_col + len(letters)

        # Matrix dimension
        self._n_row, self._n_col = self._compute_matrix_dimension(self._features_mapping)

        # Matrix
        self._features_matrix = csr_matrix((self._n_row, self._n_col))

        # Generate the matrix
        self._generate_matrix()
    # end __init__

    ############################################################
    # Override
    ############################################################

    # Get matrix
    def __call__(self, matrix_format='csr', to_array=False):
        """
        Get the matrix
        :param matrix_format: Matrix's format (csr, numpy, tensor)
        :return: The matrix
        """
        if matrix_format == 'csr':
            return self._features_matrix
        elif matrix_format == 'numpy':
            return self._features_matrix.todense()
        elif matrix_format == 'tensor':
            if to_array:
                m = self._features_matrix.toarray()
            else:
                m = self._features_matrix
            # end if
            h = int(m.shape[0])
            w = int(m.shape[1])
            x = torch.FloatTensor(1, h, w).zero_()
            for i in range(h):
                for j in range(w):
                    x[0, i, j] = m[i, j]
                # end for
            # end for
            return x
        # end if
    # end __call__

    ############################################################
    # Private
    ############################################################

    # Compute matrix dimension
    def _compute_matrix_dimension(self, features_mapping):
        """
        Compute matrix dimension
        :param features_mapping:
        :return:
        """
        # Row dimension
        n_row = len(self._letters) + 1

        # Grams
        n_col = 0
        if 'grams' in features_mapping:
            n_col += len(self._letters)
        # end if

        # End grams
        if 'end_grams' in features_mapping:
            n_col += len(self._letters)
        # end if

        # First grams
        if 'first_grams' in features_mapping:
            n_col += len(self._letters)
        # end if

        # Punctuations
        if 'punctuations' in features_mapping:
            n_col += 1
        # end if

        return n_row, n_col
    # end _compute_matrix_dimension

    # Letters to row position
    def _letter_to_position(self, letter):
        """
        Letters to row position
        :param letter:
        :return:
        """
        try:
            pos = self._letters.index(letter)
            return pos
        except ValueError:
            sys.stderr.write(u"Can not find index for {}, exiting\n".format(letter))
            exit()
        # end try
    # end

    # Dictionary sum
    def _dict_sum(self, dictionary):
        """
        Dictionary sum
        :param dictionary:
        :return:
        """
        count = 0.0
        for key in dictionary.keys():
            count += dictionary[key]
        # end for
        return float(count)
    # end _dict_sum

    # Generate gram data
    def _generate_grams_data(self, features_mapping, mapping_index, col=0):
        """
        Generate gram data
        :param features_mapping:
        :param mapping_index:
        :param col:
        :return:
        """
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)
        features_matrix = np.zeros((n_row, n_col))

        # For each gram
        for gram in features_mapping[mapping_index].keys():
            freq = features_mapping[mapping_index][gram]
            if len(gram) == 1:
                features_matrix[0, self._letter_to_position(gram)] = freq
            else:
                a = self._letter_to_position(gram[0])
                b = self._letter_to_position(gram[1])
                features_matrix[a+1, b+col] = freq
            # end if
        # end for

        return features_matrix
    # end _generate_grams_data

    # Generate end letters data
    def _generate_letters_data(self, features_mapping, mapping_index, col_pos):
        """
        Generate end letters data
        :param features_mapping:
        :param mapping_index:
        :param col_pos:
        :return:
        """
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)
        features_matrix = np.zeros((n_row, n_col))

        # For each letters
        maxi = 0
        for letter in features_mapping[mapping_index].keys():
            freq = features_mapping[mapping_index][letter]
            a = self._letter_to_position(letter[0])
            features_matrix[a + 2, col_pos] = freq
        # end for

        return features_matrix
    # end _generate_end_letters_data

    # Generate punctuations data
    def _generate_punctuations_data(self, features_mapping, mapping_index, col_pos):
        """
        Generate punctuations data
        :param features_mapping:
        :param mapping_index:
        :param col_pos:
        :return:
        """
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)
        features_matrix = np.zeros((n_row, n_col))

        # For each punctuations
        for p in features_mapping[mapping_index].keys():
            freq = features_mapping[mapping_index][p]
            a = self._punctuations.index(p)
            features_matrix[a + 2, col_pos] = freq
        # end for

        return features_matrix
    # end _generate_punctuations_data

    # Generate matrix
    def _generate_matrix(self):
        """
        Generate the matrix
        :param features_mapping:
        :return:
        """
        # Generate grams data
        if 'grams' in self._features_mapping:
            self._features_matrix += self._generate_grams_data(self._features_mapping, 'grams')
        # end if

        # Generate end grams data
        if 'end_grams' in self._features_mapping:
            self._features_matrix += self._generate_grams_data(self._features_mapping, 'end_grams', self._end_grams_col)
        # end if

        # Generate punctuation data
        if 'punctuations' in self._features_mapping:
            self._features_matrix += self._generate_punctuations_data(self._features_mapping, 'punctuations',
                                                                self._punctuations_col)
        # end if

        # Generate first grams data
        if 'first_grams' in self._features_mapping:
            self._features_matrix += self._generate_grams_data(self._features_mapping, 'first_grams', self._first_grams_col)
        # end if

        return self._features_matrix
    # end generate_matrix

# end LettersMatrix
