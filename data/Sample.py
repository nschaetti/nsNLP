#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : classifiers.TextClassifier.py
# Description : Text classifier abstract class.
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


# Sample
class Sample(object):
    """
    Sample
    """

    # Constructor
    def __init__(self, x, y):
        """

        :param x:
        :param y:
        """
        # Properties
        self._x = x
        self._y = y
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get X
    def x(self):
        """
        Get X
        :return:
        """
        return self._x
    # end x

    # Get Y
    def y(self):
        """
        Get Y
        :return:
        """
        return self._y
    # end y

# end TextClassifier
