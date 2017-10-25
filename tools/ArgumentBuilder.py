#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing Memory Project.
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
#

import argparse
import numpy as np


# Build arguments for the experiments and interpret the values
class ArgumentBuilder(object):
    """
    Build arguments for the experiments and interpret the values
    """

    # Constructor
    def __init__(self, desc):
        """
        Constructor
        :param desc:
        :param set_authors:
        """
        # Argument parser
        self._parser = argparse.ArgumentParser(description=desc)
        self._args = None
        self._arguments = dict()
    # end __init__

    #################################################
    # Public
    #################################################

    # Add argument
    def add_argument(self, command, name, help, extended, type=None, action='', required=False, default=None):
        """
        Add argument to the parser
        :param command:
        :param type:
        :param help:
        :param required:
        :param default:
        :return:
        """
        # Extended argument
        if extended:
            type=str
            default=str(default)
        # end if

        # Add argument to parser
        if action == '':
            self._parser.add_argument(command, type=type, help=help, required=required, default=default)
        else:
            self._parser.add_argument(command, action=action, help=help, required=required, default=default)
        # end if

        # Add to argument list
        self._arguments[name] = {'name': name, 'command': command, 'type': type, 'extended': extended}
    # end add_argument

    # Parse arguments
    def parse(self):
        """
        Parse arguments
        :return:
        """
        self._args = self._parser.parse_args()
    # end parse

    # Get params
    def get_params(self, params):
        """
        Get params
        :param params:
        :return:
        """
        # Params dict
        params_dict = dict()

        # Get values
        for param in params:
            if self._arguments[param]['extented']:
                params_dict[param] = self._interpret_value(self._get_value(param))
            else:
                params_dict[param] = getattr(self._args, param)
            # end if
        # end for

        return params_dict
    # end get_params

    # Get space
    def get_space(self):
        """
        Get space
        :return:
        """
        # Params dict
        params_dict = dict()

        # Get values
        for param in self._arguments.keys():
            if self._arguments[param]['extended']:
                params_dict[param] = self._interpret_value(self._get_value(param))
            # end if
        # end for

        return params_dict
    # end get_space

    #################################################
    # Override
    #################################################

    # Get attribute
    def __getattr__(self, item):
        """
        Get attribute
        :param item:
        :return:
        """
        if self._arguments[item]['extended']:
            return self._interpret_value(self._get_value(item))
        else:
            return self._get_value(item)
        # end if
    # end __getattr__

    #################################################
    # Private
    #################################################

    # Get argument's value
    def _get_value(self, param):
        """
        Get argument's value
        :param param:
        :return:
        """
        return getattr(self._args, param)
    # end get_value

    # Interpet value
    def _interpret_value(self, value):
        """
        Interpret parameter value
        :param value:
        :return:
        """
        # Value type
        value_type = 'numeric'

        # Values array
        values = np.array([])
        values_str = list()

        # Split for addition
        additions = value.split(u'+')

        # For each addition
        for add in additions:
            try:
                if ',' in add:
                    # Split by params
                    params = add.split(',')

                    # Params
                    start = float(params[0])
                    end = float(params[1]) + 0.00001
                    step = float(params[2])

                    # Add values
                    values = np.append(values, np.arange(start, end, step))
                else:
                    # Add value
                    values = np.append(values, np.array([float(add)]))
                # end if
                value_type = 'numeric'
            except ValueError:
                # Split by combinaison
                combs = add.split('*')

                # Combination elements
                comb_parts = list()

                # For each combinaison elements
                for comb in combs:
                    # Split by params
                    comb_parts.append(comb.split(','))
                # end for

                # Add to values
                values_str.append(comb_parts)

                # Value type
                value_type = 'str'
            # end try
        # end for

        if value_type == 'numeric':
            return np.sort(values)
        else:
            return values_str
        # end if
    # end _interpret_value

# end ArgumentBuilder
