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


# Metrics
class Metrics(object):
    """
    Metrics
    """

    ##############################################
    # Static
    ##############################################

    # Success rate
    @staticmethod
    def success_rate(model, sample_set):
        """
        Success rate
        :param sample_set:
        :return:
        """
        successes = 0.0
        count = 0.0

        # For each sample
        for sample in sample_set:
            # Prediction
            prediction, _ = model(sample)

            # Right?
            if prediction == sample.y():
                successes += 1.0
            # end if
        # end for

        return successes / count
    # end success_rate

# end Metrics
