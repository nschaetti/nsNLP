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
from .Tokenizer import Tokenizer
from nltk.tokenize import TweetTokenizer


# NLTK tokenizer
class NLTKTweetTokenizer(Tokenizer):
    """
    NLTK tokenizer
    """

    # Constructor
    def __init__(self, lang='english'):
        """
        Constructor
        """
        super(NLTKTweetTokenizer, self).__init__(lang, original=False)
        self._tknzr = TweetTokenizer()
    # end __init__

    ###########################################
    # Override
    ###########################################

    # Transform text to tokens
    def __call__(self, text):
        """
        Transform text to tokens
        :param text:
        :return:
        """
        return self._tknzr.tokenize(text)
    # end __call__

# end Tokenizer
