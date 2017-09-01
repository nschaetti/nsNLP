#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : statistical_models.SLTextClassifier.py
# Description : Statistical language text classifier.
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
import spacy
from sys import getsizeof
import decimal
from nsNLP.classifiers.TextClassifier import TextClassifier


# Naive Bayes classifier
class NaiveBayesClassifier(TextClassifier):
    """
    Naive Bayes Classifer
    """

    # Constructor
    def __init__(self, classes, smoothing, smoothing_param):
        """
        Constructor
        :param classes:
        :param smoothing:
        :param smoothing_param:
        """
        # Class super class
        super(NaiveBayesClassifier, self).__init__(classes=classes)

        # Properties
        self._classes = classes
        self._n_classes = classes
        self._smoothing = smoothing
        self._smoothing_param = smoothing_param

        # Initialize
        self._init()
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get name
    def name(self):
        """
        Get name
        :return:
        """
        return u"Naive Bayes classifier with {} smoothing = {}".format(self._smoothing, self._smoothing_param)
    # end name

    # Train the model
    def train(self, x, c, verbose=False):
        """
        Train
        :param x: Example's inputs
        :param c: Example's outputs
        :param verbose: Verbosity
        """
        # For each token
        for token in x.keys():
            # Count
            token_count = x[token]

            # Filtering
            token = token.replace(u"\n", u"")
            token = token.replace(u"\t", u"")
            token = token.replace(u"\r", u"")

            # Add to conditional prob.
            try:
                self._p_fi_c[c][token] += token_count
            except KeyError:
                self._p_fi_c[c][token] = token_count
            # end try

            # Add to collection prob
            try:
                self._p_fi[token] += token_count
            except KeyError:
                self._p_fi[token] = token_count
            # end try

            # Add class prob
            self._p_c[c] += token_count

            # Add total token count
            self._n_tokens += token_count
        # end for
    # end train

    # Get token count
    def get_token_count(self):
        """
        Get token count
        :return:
        """
        return self._n_tokens
    # end get_token_count

    ##############################################
    # Override
    ##############################################

    # Get token probability
    def __getitem__(self, item):
        """
        Get token probability
        :param item:
        :return:
        """
        pass
    # end __getitem__

    # To unicode
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"NaiveBayesClassifier(n_classes={}, n_tokens={}, mem_size={}o, " \
               u"token_counters_mem_size={} Go, class_counters_mem_size={} Go, n_total_token={})" \
            .format(self._n_classes, self.get_token_count(),
                    getsizeof(self), round(getsizeof(self._p_fi_c) / 1073741824.0, 4),
                    round(getsizeof(self._p_c) / 1073741824.0, 4), self._n_tokens)
    # end __str__

    ##############################################
    # Private
    ##############################################

    # Finalize the training
    def _finalize_training(self, verbose=False):
        """
        Finalize training.
        :param verbose: Verbosity
        """
        # Finalize P(Fi = fi)
        for c in self._classes:
            # Classes probs
            self._p_c[c] /= self._n_tokens

            # For each tokens
            for token in self._p_fi_c[c]:
                self._p_fi_c[c][token] = (self._p_fi_c[c][token] / self._n_tokens) / self._p_c[c]
            # end for
        # end for

        # Collecton probs
        for token in self._p_fi:
            self._p_fi[token] /= self._n_tokens
        # end for
    # end _finalize_training

    # Classify a document
    def _classify(self, x):
        """
        Classify a document.
        :param x: Document's text.
        :return: A tuple with found class and values per classes.
        """
        # Class probs
        classes_probabilities = dict()
        for c in self._classes:
            classes_probabilities[c] = decimal.Decimal(1.0)
        # end for

        # Document length
        doc_len = 0.0
        for token in x:
            doc_len += x[token]
        # end for

        # For each classes
        for c in self._classes:
            # For each tokens
            for token in x.keys():
                # Token count
                token_count = decimal.Decimal(x[token])

                # Prob Fi knowning c
                try:
                    p_fi_c = decimal.Decimal(self._p_fi_c[c][token])
                except KeyError:
                    p_fi_c = decimal.Decimal(0.0)
                # end try

                # Prob Fi
                try:
                    p_fi = decimal.Decimal(self._p_fi[token])
                except KeyError:
                    p_fi = decimal.Decimal(0.0)
                # end try

                # Token prob exists
                if p_fi > 0:
                    p_fi_c = NaiveBayesClassifier.smooth(self._smoothing, p_fi_c, p_fi, doc_len, self._smoothing_param)
                    classes_probabilities[c] *= decimal.Decimal(p_fi_c) * token_count
                # end if
            # end for
        # end for

        # Get max
        max = decimal.Decimal(0.0)
        max_class = None
        for c in self._classes:
            c_probs = decimal.Decimal(self._p_c[c]) * classes_probabilities[c]
            if c_probs > max:
                max = c_probs
                max_class = c
            # end if
        # end for

        return max_class, classes_probabilities
    # end _classify

    # Reset the classifier
    def _reset_model(self):
        """
        Reset the classifier
        """
        self._init()
    # end reset

    # Init classifier
    def _init(self):
        """
        Init classifier
        :return:
        """
        # Conditional probabilities
        self._p_fi_c = dict()
        for c in self._classes:
            self._p_fi_c[c] = dict()
        # end for

        # Class probabilities
        self._p_c = dict()
        for c in self._classes:
            self._p_c[c] = 0.0
        # end for

        # Collection_prob
        self._p_fi = dict()

        # Total count
        self._n_tokens = 0
    # end _init

    ##############################################
    # Static
    ##############################################

    # Dirichlet prior smoothing function
    @staticmethod
    def smooth_dirichlet_prior(doc_prob, col_prob, doc_length, mu):
        """
        Dirichlet prior smoothing function
        :param doc_prob:
        :param col_prob:
        :param doc_length:
        :param mu:
        :return:
        """
        return (float(doc_length) / (float(doc_length) + float(mu))) * doc_prob + \
               (float(mu) / (float(mu) + float(doc_length))) * col_prob
    # end smooth

    # Jelinek Mercer smoothing function
    @staticmethod
    def smooth_jelinek_mercer(doc_prob, col_prob, param_lambda):
        """
        Jelinek Mercer smoothing function
        :param col_prob:
        :param param_lambda:
        :return:
        """
        return (1.0 - param_lambda) * doc_prob + param_lambda * col_prob
    # end smooth

    # Smoothing function
    @staticmethod
    def smooth(smooth_algo, doc_prob, col_prob, doc_length, param):
        """
        Smoothing function
        :param smooth_algo: Algo type
        :param doc_prob:
        :param col_prob:
        :param doc_length:
        :param param:
        :return:
        """
        if smooth_algo == "dp":
            return NaiveBayesClassifier.smooth_dirichlet_prior(doc_prob, col_prob, doc_length, param)
        else:
            return NaiveBayesClassifier.smooth_jelinek_mercer(doc_prob, col_prob, param)
        # end if
    # end smooth

# end SLTextClassifier
