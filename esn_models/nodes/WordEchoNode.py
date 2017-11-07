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
class WordEchoNode(Oger.nodes.ReservoirNode):
    """
    A EchoWord node
    """

    # Variables
    direction = ""

    # Constructor
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
                 nonlin_func=np.tanh, reset_states=True, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
                 w_in=None, w=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 set_initial_state=False, my_initial_state=None, use_sparse_matrix=False, direction='both'):
        """
        Constructor
        :param input_dim:
        :param output_dim:
        :param spectral_radius:
        :param nonlin_func:
        :param reset_states:
        :param bias_scaling:
        :param input_scaling:
        :param dtype:
        :param _instance:
        :param w_in:
        :param w:
        :param w_bias:
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param set_initial_state:
        :param my_initial_state:
        :param use_sparse_matrix:
        :param direction:
        """
        # Call upper class
        super(WordEchoNode, self).__init__(input_dim=input_dim, output_dim=output_dim,
                                           spectral_radius=spectral_radius, nonlin_func=nonlin_func,
                                           reset_states=reset_states, bias_scaling=bias_scaling,
                                           input_scaling=input_scaling, dtype=dtype, _instance=_instance,
                                           w_in=w_in, w=w, w_bias=w_bias, sparsity=sparsity, input_set=input_set,
                                           w_sparsity=w_sparsity, set_initial_state=set_initial_state,
                                           my_initial_state=my_initial_state, use_sparse_matrix=use_sparse_matrix)

        # Properties
        self.direction = direction
    # end __init__

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
            lr_states = super(Oger.nodes.ReservoirNode, self)._execute(x)[:-1, :]
        # end if

        # Compute the states from right to left
        if self.direction == 'both' or self.direction == 'rl':
            reversed_inputs = self._flip_matrix(x)
            rl_states = super(Oger.nodes.ReservoirNode, self)._execute(reversed_inputs)[:-1, :]
        # end if

        # Return
        if self.direction == 'both':
            return lr_states, rl_states
        elif self.direction == 'lr':
            return lr_states
        else:
            return rl_states
        # end if
    # _execute

    # Flip sparse matrix
    def _flip_matrix(self, m):
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


# Parallel Word Echo node
class ParallelWordEchoNode(mdp.parallel.ParallelExtensionNode, WordEchoNode):
    """
    Parallel Word Echo node
    """

    # Fork
    def _fork(self):
        """
        Fork
        :return:
        """
        return self._default_fork()
    # end _fork

# end ParallelWordEchoNode

