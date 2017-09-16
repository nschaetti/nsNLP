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


# Bag of Grams
class BagOfGrams(object):
    """
    Bag of Grams
    """

    # Constructor
    def __init__(self, uppercase=False):
        """
        Constructor
        :param text:
        """
        self._uppercase = uppercase
        self._bows = list()
    # end __init__

    #########################################
    # Public
    #########################################

    # Add features
    def add(self, bow):
        """
        Add features
        :param bow:
        :return:
        """
        self._bows.append(bow)
    # end add

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

        # For each features
        for bow in self._bows:
            # Get features
            features = bow(tokens)

            # For each token
            for token in features.keys():
                try:
                    voc_count[token] += features[token]
                except KeyError:
                    voc_count[token] = features[token]
                # end
            # end for
        # end for

        return voc_count
    # end __call__

    #########################################
    # Private
    #########################################

# end BagOfGrams
