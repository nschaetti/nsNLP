#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the nsNLP toolbox.
# The RnsNLP toolbox is a set of free software:
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

# Import packages
import numpy as np
import Oger
import mdp
from datetime import datetime
from sys import getsizeof
from nsNLP.generator.TextGenerator import TextGenerator
import matplotlib.pyplot as plt
from decimal import *
import logging
import pickle
import os
from converters.PosConverter import PosConverter
from converters.TagConverter import TagConverter
from converters.WVConverter import WVConverter
from converters.FuncWordConverter import FuncWordConverter
from converters.OneHotConverter import OneHotConverter
from converters.LetterConverter import LetterConverter
import converters.JoinConverter


# ESN generator model
class ESNTextGenerator(TextGenerator):
    """
    ESN generator model
    """
    pass
# end ESNTextGenerator
