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
    def __init__(self, alphabet, uppercase=False, n_gram=1, multi=10, tokenizer=None, start_grams=False, end_grams=False):
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
        self._multi = multi
        self._start_grams = start_grams
        self._end_grams = end_grams
        self._tokenizer = tokenizer

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
        # Size factor
        size_factor = 1

        # Start grams
        if self._start_grams:
            size_factor += 1
        # end if

        # End grams
        if self._end_grams:
            size_factor += 1
        # end if

        # Tensor
        gram_tensor = None
        if self._n_gram == 1:
            gram_tensor = torch.zeros(1, self._n_chars * size_factor)
        elif self._n_gram == 2:
            gram_tensor = torch.zeros(1, self._n_chars * size_factor, self._n_chars + 1)
        elif self._n_gram == 3:
            gram_tensor = torch.zeros(1, self._n_chars * size_factor, self._n_chars + 1, self._n_chars + 1)
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

        # Start characters
        if self._start_grams:
            self._compute_position_1gram(gram_tensor, text, 'start', self._n_chars, normalize)
        # end if

        # Start 2-grams
        if self._start_grams and self._n_gram >= 2:
            self._compute_position_2gram(gram_tensor, text, 'start', self._n_chars, normalize)
        # end if

        # End characters
        if self._end_grams:
            self._compute_position_1gram\
            (
                gram_tensor,
                text,
                'end',
                self._n_chars if not self._start_grams else self._n_chars * 2,
                normalize
            )
        # end if

        # End grams
        if self._end_grams and self._n_gram >= 2:
            self._compute_position_2gram\
            (
                gram_tensor,
                text,
                'end',
                self._n_chars if not self._start_grams else self._n_chars * 2,
                normalize
            )
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
                tensor[0, :, 0] /= tensor[0, :, 0].max()
                tensor[0, :, 0] *= self._multi
            elif self._n_gram == 3:
                tensor[0, :, 0, 0] /= total
                tensor[0, :, 0, 0] /= tensor[0, :, 0, 0].max()
                tensor[0, :, 0, 0] *= self._multi
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
                tensor[0, :, 1:] /= tensor[0, :, 1:].max()
                tensor[0, :, 1:] *= self._multi
            elif self._n_gram == 3:
                tensor[0, :, 1:, 0] /= total
                tensor[0, :, 1:, 0] /= tensor[0, :, 1:, 0]
                tensor[0, :, 1:, 0] *= self._multi
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
            tensor[0, :, 1:, 1:] /= tensor[0, :, 1:, 1:].max()
            tensor[0, :, 1:, 1:] *= self._multi
        # end if
    # end _compute_3gram

    # Compute position grams
    def _compute_position_1gram(self, tensor, text, gram_type, start_pos, normalize=True):
        """
        Compute position 1grams
        :param tensor:
        :param text:
        :param gram_type:
        :param start_pos:
        :param normalize:
        :return:
        """
        # Total
        total = 0.0

        # Check tokenizer
        if self._tokenizer is None:
            raise Exception(u"I need a tokenizer!")
        # end if

        # Gram positin
        gram_pos1 = 0
        if gram_type == 'end':
            gram_pos1 = -1
        # end if

        # For each token
        for token in self._tokenizer(text):
            # Length
            if len(token) > 0:
                # Index
                char_index1 = self._get_char_index(token[gram_pos1])

                # Set
                if self._n_gram == 1:
                    tensor[0, start_pos + char_index1] += 1.0
                if self._n_gram == 2:
                    tensor[0, start_pos + char_index1, 0] += 1.0
                else:
                    tensor[0, start_pos + char_index1, 0, 0] += 1.0
                # end if

                # Total
                total += 1.0
            # end if
        # end for

        # Normalize
        if normalize:
            if self._n_gram == 1:
                tensor[0, start_pos:start_pos + self._n_chars] /= total
            if self._n_gram == 2:
                tensor[0, start_pos:start_pos + self._n_chars, 0] /= total
                max = tensor[0, start_pos:start_pos + self._n_chars, 0].max()
                tensor[0, start_pos:start_pos + self._n_chars, 0] /= max
                tensor[0, start_pos:start_pos + self._n_chars, 0] *= self._multi
            elif self._n_gram == 3:
                tensor[0, start_pos:start_pos + self._n_chars, 0, 0] /= total
                max = tensor[0, start_pos:start_pos + self._n_chars, 0, 0].max()
                tensor[0, start_pos:start_pos + self._n_chars, 0, 0] /= max
                tensor[0, start_pos:start_pos + self._n_chars, 0, 0] *= self._multi
            # end if
        # end if
    # end _compute_position_1gram

    # Compute position grams
    def _compute_position_2gram(self, tensor, text, gram_type, start_pos, normalize=True):
        """
        Compute position grams
        :param tensor:
        :param text:
        :param start_pos:
        :param normalize:
        :return:
        """
        # Total
        total = 0.0

        # Check tokenizer
        if self._tokenizer is None:
            raise Exception(u"I need a tokenizer!")
        # end if

        # Gram positin
        gram_pos1 = 0
        gram_pos2 = 1
        if gram_type == 'end':
            gram_pos1 = -2
            gram_pos2 = -1
        # end if

        # For each token
        for token in self._tokenizer(text):
            # Length
            if len(token) > 1:
                # Index
                char_index1 = self._get_char_index(token[gram_pos1])
                char_index2 = self._get_char_index(token[gram_pos2])

                # Add
                char_index2 = char_index2 + 1 if char_index2 != -1 else -1

                # Set
                if self._n_gram == 2:
                    tensor[0, start_pos + char_index1, char_index2] += 1.0
                else:
                    tensor[0, start_pos + char_index1, char_index2, 0] += 1.0
                # end if

                # Total
                total += 1.0
            # end if
        # end for

        # Normalize
        if normalize:
            if self._n_gram == 2:
                tensor[0, start_pos:start_pos+self._n_chars, 1:] /= total
                max = tensor[0, start_pos:start_pos+self._n_chars, 1:].max()
                tensor[0, start_pos:start_pos + self._n_chars, 1:] /= max
                tensor[0, start_pos:start_pos + self._n_chars, 1:] *= self._multi
            elif self._n_gram == 3:
                tensor[0, start_pos:start_pos+self._n_chars, 1:, 0] /= total
                max = tensor[0, start_pos:start_pos+self._n_chars, 1:, 0].max()
                tensor[0, start_pos:start_pos + self._n_chars, 1:, 0] /= max
                tensor[0, start_pos:start_pos + self._n_chars, 1:, 0] *= self._multi
            # end if
        # end if
    # end _compute_position_gram

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