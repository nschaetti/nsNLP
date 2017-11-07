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

# Imports
import mdp
import Oger
import numpy as np
import scipy.sparse


#########################################
# Context State node
#########################################
class ContextStateNode(mdp.Node):
    """
    Context State node
    """

    # Constructor
    def __init__(self, input_dim=100, state_gram=1, dtype='float64'):
        super(ContextStateNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self._state_gram = state_gram
    # end __init__

    # This node is not trainable
    def is_trainable(self):
        """
        This node is not trainable
        :return:
        """
        return False
    # end is_trainable

    # Execute this node
    def _execute(self, x):
        """
        Execute this node.
        :param x:
        :return:
        """
        pass
    # end _execute

# end ContextStateNode
