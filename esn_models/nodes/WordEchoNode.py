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
# Word Echo node
#########################################
class WordEchoNode(Oger.nodes.LeakyReservoirNode):
    """
    A EchoWord node
    """

    # Variables
    direction = ""

    # Constructor
    def __init__(self, output_dim, direction='both', *args, **kwargs):
        """
        Constructor
        :param direction:
        :param args:
        :param kwargs:
        """
        # Call upper class
        super(WordEchoNode, self).__init__(output_dim=output_dim, *args, **kwargs)

        # Properties
        self.direction = direction

        # Output states size
        if direction == 'both':
            self.output_states_size = output_dim * 2
        else:
            self.output_states_size = output_dim
        # end if
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

    # Executes simulation with input vector x
    def _execute(self, x):
        """
        Execute simulation with input vector x
        :param x:
        :return:
        """
        # Init
        lr_states = np.array([])
        rl_states = np.array([])

        # Compute the states from left to right
        if self.direction == 'both' or self.direction == 'lr':
            lr_states = super(WordEchoNode, self)._execute(x)
        # end if

        # Compute the states from right to left
        if self.direction == 'both' or self.direction == 'rl':
            reversed_inputs = WordEchoNode.flip_matrix(x)
            rl_states = np.flip(super(WordEchoNode, self)._execute(reversed_inputs), axis=0)
        # end if

        # Return
        if self.direction == 'lr':
            return lr_states
        elif self.direction == 'rl':
            return rl_states
        else:
            join_states = np.hstack((lr_states, rl_states))
            return join_states
        # end if
    # _execute

    ###############################################
    # Static
    ###############################################

    # Flip sparse matrix
    @staticmethod
    def flip_matrix(m):
        """
        Flip sparse matrix
        :param m:
        :return:
        """
        # New CSR
        m_flip = scipy.sparse.csr_matrix((m.shape[0], m.shape[1]))

        # Go backward
        j = 0
        for i in np.arange(m.shape[0] - 1, -1, -1):
            m_flip[j, :] = m[i, :]
            j += 1
        # end for

        return m_flip
    # end _flip_matrix

# end

