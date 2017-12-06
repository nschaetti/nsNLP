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
import numpy as np


# Multinomial sampling node
class MultinomialSamplingNode(mdp.Node):
    """
    Multinomial sampling node
    """

    # Constructor
    def __init__(self, dtype='float64'):
        super(MultinomialSamplingNode, self).__init__(input_dim=None, dtype=dtype)
    # end __init__

    ###############################################
    # Public
    ###############################################

    # This node is not trainable
    def is_trainable(self):
        """
        This node is not trainable
        :return:
        """
        return False
    # end is_trainable

    ###############################################
    # Private
    ###############################################

    # Execute this node
    def _execute(self, x):
        """
        Execute this node.
        :param x:
        :return:
        """
        # N. samples
        n_samples = x.shape[0]

        # Output size
        output_size = x.shape[1]

        # One-hot output
        hot_output = np.zeros((n_samples, output_size))

        # For each output
        for i in n_samples:
            index = self._multinomial_sample(x[i, :])
            hot_output[i, index] = 1.0
        # end for

        return hot_output
    # end _execute

# end

