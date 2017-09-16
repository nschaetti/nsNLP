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


# Bag of 2 grams
class BagOf2Grams(object):
    """
    Bag of 2-grams
    """

    # Constructor
    def __init__(self, uppercase=False):
        """
        Constructor
        :param text:
        """
        self._uppercase = uppercase
    # end __init__

    #########################################
    # Public
    #########################################

    #########################################
    # Override
    #########################################

    # Call
    def __call__(self, tokens):
        """
        Call
        :return:
        """
        # Vocabulary
        voc_count = dict()

        # Preceding token
        preceding_token = None

        # For each tokens
        for token in tokens:
            # Uppercase?
            if not self._uppercase:
                token = token.lower()
            # end if

            if preceding_token is None:
                preceding_token = token
            else:
                # Bigram
                bigram = preceding_token + u" " + token

                # Add
                try:
                    voc_count[bigram] += 1.0
                except KeyError:
                    voc_count[bigram] = 1.0
                # end try

                # Preceding token
                preceding_token = token
            # end if
        # end for

        return voc_count
    # end __call__

    #########################################
    # Private
    #########################################

# end BagOfWords
