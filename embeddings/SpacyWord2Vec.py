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
import spacy
from Word2Vec import Word2Vec


# Transform word to a vector
class SpacyWord2Vec(Word2Vec):
    """
    Transform word to a vector
    """

    # Constructor
    def __init__(self, lang):
        """
        Constructor
        :param lang: Language
        """
        super(SpacyWord2Vec, self).__init__(lang)
        self._nlp = spacy.load(lang)
    # end __init__

    ############################################
    # Override
    ############################################

    # Transform a list of tokens
    def __call__(self, tokens):
        """
        Transform a list of tokens
        :param tokens:
        :return:
        """
        vectors = list()
        for token in tokens:
            vectors.append(self[token])
        # end for
        return vectors
    # end __call__

    # Override get item
    def __getitem__(self, item):
        """
        Override get item
        :param item:
        :return:
        """
        return self._nlp(item).vector
    # end __getitem__

# end Word2Vec
