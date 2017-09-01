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
import spacy


# Bag of Words
class BagOfWords(object):
    """
    Bag of Words
    """

    # Constructor
    def __init__(self, lang, uppercase=False):
        """
        Constructor
        :param text:
        """
        self._lang = lang
        self._uppercase = uppercase
        self._voc_count = dict()
    # end __init__

    #########################################
    # Public
    #########################################

    #########################################
    # Override
    #########################################

    # Call
    def __call__(self, x):
        """
        Call
        :return:
        """
        # Tokens
        tokens = spacy.load(self._lang)(x)

        # For each tokens
        for token in tokens:
            # Token text
            token_text = token.text

            # Add
            try:
                self._voc_count[token_text] += 1.0
            except KeyError:
                self._voc_count[token_text] = 1.0
            # end try
        # end for

        return self._voc_count
    # end __call__

    #########################################
    # Private
    #########################################

# end BagOfWords
