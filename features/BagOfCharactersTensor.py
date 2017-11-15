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
import torch


# Bag of characters tensor
class BagOfCharactersTensor(object):
    """
    Bag of characters tensor
    """

    # Constructor
    def __init__(self, alphabet, uppercase=False, n_gram=1):
        """
        Constructor
        :param n_gram:
        """
        # Variables
        self._alphabet = alphabet
        self._uppercase = uppercase
        self._n_gram = n_gram
        self._n_chars = len(alphabet) + 1
        self.chars = list()
        self.char_count = dict()

        # Char to index
        self._char2index = dict()
        for index, c in enumerate(alphabet):
            self._char2index[c] = index
        # end for
    # end __init__

    #########################################
    # Override
    #########################################

    # Call
    def __call__(self, text, normalize=True):
        """
        Call
        :return:
        """
        # Tensor
        gram_tensor = None
        if self._n_gram == 1:
            gram_tensor = torch.zeros(1, self._n_chars)
        elif self._n_gram == 2:
            gram_tensor = torch.zeros(1, self._n_chars, self._n_chars + 1)
        elif self._n_gram == 3:
            gram_tensor = torch.zeros(1, self._n_chars, self._n_chars + 1, self._n_chars + 1)
        # end if

        # Compute 1 gram
        self._compute_1gram(gram_tensor, text, normalize)

        # Compute 2 gram
        if self._n_gram >= 2:
            self._compute_2gram(gram_tensor, text, normalize)
        # end if

        # Compute 3 gram
        if self._n_gram == 3:
            self._compute_3gram(gram_tensor, text, normalize)
        # end if

        return gram_tensor
    # end __call__

    #########################################
    # Private
    #########################################

    # Compute 1-gram values
    def _compute_1gram(self, tensor, text, normalize=True):
        """
        Compute 1-gram values
        :param tensor:
        :param text:
        :return:
        """
        # Total
        total = 0.0

        # For each grams
        for i in range(len(text)):
            # Gram
            gram = text[i]

            # Index
            char_index = self._get_char_index(gram)

            # Set
            if self._n_gram == 1:
                tensor[0, char_index] += 1.0
            elif self._n_gram == 2:
                tensor[0, char_index, 0] += 1.0
            elif self._n_gram == 3:
                tensor[0, char_index, 0, 0] += 1.0
            # end if
            total += 1.0
        # end for

        # Normalize
        if normalize:
            if self._n_gram == 1:
                tensor /= total
            elif self._n_gram == 2:
                tensor[0, :, 0] /= total
            elif self._n_gram == 3:
                tensor[0, :, 0, 0] /= total
            # end if
        # end if
    # end _compute_1gram

    # Compute 2-gram values
    def _compute_2gram(self, tensor, text, normalize=True):
        """
        Compute 2-gram values
        :param tensor:
        :param text:
        :param normalize:
        :return:
        """
        # Total
        total = 0.0

        # For each grams
        for i in range(len(text)-1):
            # Gram
            gram = text[i:i+2]

            # Index
            char_index1 = self._get_char_index(gram[0])
            char_index2 = self._get_char_index(gram[1])

            # Add
            char_index2 = char_index2 + 1 if char_index2 != -1 else -1

            # Set
            if self._n_gram == 2:
                tensor[0, char_index1, char_index2] += 1.0
            elif self._n_gram == 3:
                tensor[0, 0, char_index1, char_index2, 0] += 1.0
            # end if

            total += 1.0
        # end for

        # Normalize
        if normalize:
            if self._n_gram == 2:
                tensor[0, :, 1:] /= total
            elif self._n_gram == 3:
                tensor[0, :, 1:, 0] /= total
            # end if
        # end if
    # end _compute_2gram

    # Compute 3-gram values
    def _compute_3gram(self, tensor, text, normalize=True):
        """
        Compute 3-gram values
        :param tensor:
        :param text:
        :param normalize:
        :return:
        """
        # Total
        total = 0.0

        # For each grams
        for i in range(len(text)-2):
            # Gram
            gram = text[i:i + 3]

            # Index
            char_index1 = self._get_char_index(gram[0])
            char_index2 = self._get_char_index(gram[1])
            char_index3 = self._get_char_index(gram[2])

            # Add
            char_index2 = char_index2 + 1 if char_index2 != -1 else -1
            char_index3 = char_index3 + 1 if char_index3 != -1 else -1

            # Set
            tensor[0, char_index1, char_index2, char_index3] += 1.0

            total += 1.0
        # end for

        # Normalize
        if normalize:
            tensor[0, :, 1:, 1:] /= total
        # end if
    # end _compute_3gram

    # Get char index
    def _get_char_index(self, c):
        """
        Get char index
        :param c:
        :return:
        """
        try:
            return self._char2index[c]
        except KeyError:
            return -1
        # end try
    # end _get_char_index

# end BagOfCharactersTensor