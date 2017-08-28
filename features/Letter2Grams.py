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


# 2 Grams of letters
class Letter2Grams(object):
    """
    Frequencies of 2-grams of letters
    """

    # Constructor
    def __init__(self, text, alphabet, punc):
        """
        Constructor
        :param text:
        """
        self._alphabet = alphabet
        self._punc = punc
        self._text = text
    # end __init__

    #########################################
    # Public
    #########################################

    # Get beginning letters
    def get_beginning_letters(self, uppercase=True):
        """
        Get beginning letters
        :param uppercase:
        :return:
        """
        return self._get_positional_letters(0, uppercase=uppercase)
    # end get_beginning_letters

    # Get ending letters
    def get_ending_letters(self, uppercase=True):
        """
        Get beginning letters
        :param uppercase:
        :return:
        """
        return self._get_positional_letters(-1, uppercase=uppercase)
    # end get_beginning_letters

    # Get beginning 2grams
    def get_beginning_2grams(self, uppercase=True):
        """
        Get beggining 2grams
        :param uppercase:
        :return:
        """
        return self._get_positional_2grams(0, uppercase=uppercase)
    # end get_beginning_2grams

    # Get 3-grams
    def get_3grams(self, uppercase=True):
        """
        Get 3-grams
        :return:
        """
        # Cleaned text
        cleaned_text = self._clean_text(self._text)

        # Uppercase?
        if not uppercase:
            cleaned_text = cleaned_text.lower()
        # end if

        # Total
        total = 0.0

        # Result
        result = dict()

        # For each tokens
        for token in cleaned_text.split(u' '):
            if len(token) > 2:
                for i in range(len(token) - 1):
                    gram = token[i:i + 3]
                    if gram not in result.keys():
                        result[gram] = 0.0
                    # end if
                    result[gram] += 1.0
                    total += 1.0
                # end for
            elif len(token) > 0:
                gram = token
                if gram not in result.keys():
                    result[gram] = 0.0
                # end if
                result[gram] += 1.0
                total += 1.0
            # end if
        # end for

        # Normalize
        result = self._normalize(result, total)

        return result
    # end get_3grams

    # Get 2-grams
    def get_2grams(self, uppercase=True):
        """
        Get 2-grams
        :return:
        """
        # Cleaned text
        cleaned_text = self._clean_text(self._text)

        # Uppercase?
        if not uppercase:
            cleaned_text = cleaned_text.lower()
        # end if

        # Total
        total = 0.0

        # Result
        result = dict()

        # For each tokens
        for token in cleaned_text.split(u' '):
            if len(token) > 1:
                for i in range(len(token) - 1):
                    gram = token[i:i + 2]
                    if gram not in result.keys():
                        result[gram] = 0.0
                    # end if
                    result[gram] += 1.0
                    total += 1.0
                    # end for
            elif len(token) == 1:
                gram = token
                if gram not in result.keys():
                    result[gram] = 0.0
                # end if
                result[gram] += 1.0
                total += 1.0
            # end if
        # end for

        # Normalize
        result = self._normalize(result, total)

        return result
    # end get_2grams

    # Get letters
    def get_letter_frequencies(self, uppercase=True):
        """
        Get letters
        :return:
        """
        # Cleaned text
        cleaned_text = self._clean_text(self._text)

        # Uppercase?
        if not uppercase:
            cleaned_text = cleaned_text.lower()
        # end if

        # Total
        total = 0.0

        # Result
        result = dict()

        # For each tokens
        for token in cleaned_text.split(u' '):
            # For each letter
            for i in range(len(token)):
                letter = token[i]
                if letter not in result.keys():
                    result[letter] = 0.0
                # end if
                result[letter] += 1.0
                total += 1.0
                # end for
        # end for

        # Normalize
        result = self._normalize(result, total)

        return result
    # end get_letters

    #########################################
    # Private
    #########################################

    # Get position 2grams
    def _get_positional_2grams(self, pos=0, uppercase=True):
        """
        Get beginning letters
        :param uppercase:
        :return:
        """
        # Cleaned text
        cleaned_text = self._clean_text(self._text)

        # Uppercase?
        if not uppercase:
            cleaned_text = cleaned_text.lower()
        # end if

        # Total
        total = 0.0

        # Result
        result = dict()

        # For each tokens
        for token in cleaned_text.split(u' '):
            if len(token) > 1:
                gram = token[pos:pos+2]
                if gram not in self._punc:
                    if gram not in result.keys():
                        result[gram] = 0.0
                    # end if
                    result[gram] += 1.0
                    total += 1.0
                # end if
            # end if
        # end for

        # Normalize
        result = self._normalize(result, total)

        return result
    # end _get_positional_letters

    # Get beginning letters
    def _get_positional_letters(self, pos=0, uppercase=True):
        """
        Get beginning letters
        :param uppercase:
        :return:
        """
        # Cleaned text
        cleaned_text = self._clean_text(self._text)

        # Uppercase?
        if not uppercase:
            cleaned_text = cleaned_text.lower()
        # end if

        # Total
        total = 0.0

        # Result
        result = dict()

        # For each tokens
        for token in cleaned_text.split(u' '):
            if len(token) > 0:
                if token[pos] not in self._punc:
                    if token[pos] not in result.keys():
                        result[token[pos]] = 0.0
                    # end if
                    result[token[pos]] += 1.0
                    total += 1.0
                    # end if
                    # end if
        # end for

        # Normalize
        result = self._normalize(result, total)

        return result
    # end _get_positional_letters

    # Normalize
    def _normalize(self, result, total):
        """
        Normalize
        :param result:
        :return:
        """
        # Normalize
        for gram in result.keys():
            result[gram] /= total
        # end for

        # All results
        freqs = np.zeros(len(result.keys()))
        for index, gram in enumerate(result.keys()):
            freqs[index] = result[gram]
        # end for

        # Normalize
        for gram in result.keys():
            result[gram] /= np.max(freqs)
        # end for

        return result
    # end _normalize

    # Clean text
    def _clean_text(self, text):
        """
        Clean text
        :param text:
        :return:
        """
        # Cleaned text
        cleaned_text = ""

        # Clean text
        for i in range(len(text)):
            letter = text[i]
            if letter in self._alphabet or letter in self._punc:
                cleaned_text += letter
            else:
                cleaned_text += u" "
            # end if
        # end for

        # Tokenize punctuations
        for p in self._punc:
            cleaned_text = cleaned_text.replace(p, u" " + p + u" ")
        # end for

        # No multiple space
        for i in range(20):
            cleaned_text = cleaned_text.replace(u"  ", u" ")
        # end for

        return cleaned_text
    # end _clean_text

# end Letter2Grams
