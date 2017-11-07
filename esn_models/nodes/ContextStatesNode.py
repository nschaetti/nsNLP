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
    def __init__(self, direction, input_size, state_gram=1, dtype='float64'):
        super(ContextStateNode, self).__init__(input_dim=None, dtype=dtype)
        # Variables
        self._input_size = input_size
        self._state_gram = state_gram
        self._direction = direction
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
        # States
        lr_states = np.array([])
        rl_states = np.array([])

        # Length
        n_states = x.shape[0]

        # States size
        x_size = x.shape[1]

        # Gram size
        side_size = 1

        # Reservoir size
        reservoir_size = x_size

        # Extract state
        if self._direction == 'both':
            reservoir_size = int(x_size / 2.0)
            lr_states = x[:, :reservoir_size]
            rl_states = x[:, reservoir_size:]
            side_size = 2
        elif self._direction == 'lr':
            lr_states = x
        elif self._direction == 'rl':
            rl_states = x
        # end if

        # Context states
        context_states = np.zeros((n_states, reservoir_size * self._state_gram * side_size))

        # For each position
        for index in range(n_states):
            # Direction
            if self._direction == 'both':
                left_states = self._fill_states(lr_states[index - self._state_gram:index].flatten(), reservoir_size,
                                                'left')
                right_states = self._fill_states(rl_states[index + 1:index + self._state_gram + 1].flatten(),
                                                 reservoir_size, 'right')
                context = np.hstack((left_states, right_states))
            elif self._direction == 'lr':
                context = self._fill_states(lr_states[index - self._state_gram:index].flatten(), reservoir_size, 'left')
            elif self._direction == 'rl':
                context = self._fill_states(rl_states[index + 1:index + self._state_gram + 1].flatten(), reservoir_size,
                                            'right')
            # end if

            # Set state
            context_states[index, :] = context
        # end for

        return context_states
    # end _execute

    # Fill states
    def _fill_states(self, x, reservoir_size, side):
        """
        Fill states
        :param x:
        :param direction:
        :return:
        """
        # State size
        state_size = int(reservoir_size * self._state_gram)

        # States
        states = np.zeros(state_size)

        # Direction
        if side == 'left':
            states[state_size-x.shape[0]:] = x
        elif side == 'right':
            states[:x.shape[0]] = x
        # end if

        return states
    # end _fill_states

# end ContextStateNode
