#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
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

# Imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pylab as plt
import codecs
import unicodecsv as csv


# Transform word to a vector
class Embeddings(object):
    """
    Transform word to a vector
    """

    # Constructor
    def __init__(self, size):
        """
        Constructor
        :param lang: Language
        """
        # Globals
        self._word2vec = dict()
        self._properties = dict()
        self._size = int(size)
        self._property_list = list()
    # end __init__

    ############################################
    # Properties
    ############################################

    # Embeddings size
    @property
    def size(self):
        """
        Embeddings size
        :return:
        """
        return self._size
    # end size

    # Vocabulary size
    @property
    def voc_size(self):
        """
        Vocabulary size
        :return:
        """
        return len(self._word2vec.keys())
    # end voc_size

    # List of words
    @property
    def voc(self):
        """
        List of words
        :return:
        """
        return self._word2vec.keys()
    # end voc

    ############################################
    # Public
    ############################################

    # Save list of words
    def wordlist(self, csv_file, info=u""):
        """
        Save list of words
        :param csv_file:
        :param info:
        :return:
        """
        # File
        with open(csv_file, 'wb') as f:
            # CSV writer
            csv_writer = csv.writer(f, )

            # Write header
            csv_writer.writerow(['word'] + self._property_list)

            # For each words
            for word in self.voc:
                word_desc = list()
                word_desc.append(word)
                for prop in self._property_list:
                    try:
                        word_desc.append(self._properties[word][prop])
                    except KeyError:
                        word_desc.append(u"")
                    # end try
                # end for
                csv_writer.writerow(word_desc)
            # end for
        # end with

        # Save info
        if info != u"":
            with codecs.open(csv_file + u".txt", 'w', encoding='utf-8') as f:
                f.write(info)
            # end with
        # end if
    # end wordlist

    # Save the image of word vectors reduced
    def wordnet(self, prop, image, n_words=100, fig_size=5000, reduction='TSNE', info=u""):
        """
        Save the image of word vectors reduced
        :param property:
        :param image:
        :param fig_size:
        :param reduction:
        :return:
        """
        # Word counts
        word_counts = list()

        # Select words with highest property value
        for word in self.voc:
             word_counts.append((word, self.get(word, prop)))
        # end for

        # Order list by property value
        word_counts.sort(key=lambda tup: tup[1], reverse=True)

        # Selected words
        selected_words = [i[0] for i in word_counts[:n_words]]

        # Selected word embeddings
        selected_word_embeddings = np.zeros((int(self._size), int(n_words)))

        # For each words
        selected_word_indexes = dict()
        for index, word in enumerate(selected_words):
            selected_word_embeddings[:, index] = self[word]
            selected_word_indexes[word] = index
        # end for

        # Reduce
        reduced_matrix = Embeddings.reduction(selected_word_embeddings, reduction)

        # Save figure
        Embeddings.save_figure(reduced_matrix, selected_word_indexes, image, fig_size)

        # Save info
        if info != u"":
            with codecs.open(image + u".txt", 'w', encoding='utf-8') as f:
                f.write(info)
            # end with
        # end if
    # end words_figure

    # Add a correspondence
    def add(self, word, vector):
        """
        Add a correspondance
        :param word:
        :param vector:
        :return:
        """
        self._word2vec[word] = vector
    # end add

    # Add multiple correspondences
    def adds(self, words, vectors):
        """
        Add multiple correspondences
        :param words:
        :param vectors:
        :return:
        """
        for index, word in enumerate(words):
            self._word2vec[word] = vectors[index]
        # end for
    # end adds

    # Set a property about a word
    def set(self, word, property, value):
        """
        Set a property about a word
        :param word:
        :param property:
        :param value:
        :return:
        """
        # Add to list
        if property not in self._property_list:
            self._property_list.append(property)
        # end if

        # Add dict
        if word not in self._properties:
            self._properties[word] = dict()
        # end if

        # Set properties
        self._properties[word][property] = value
    # end set

    # Get a property about a word
    def get(self, word, property):
        """
        Get a property about a word
        :param word:
        :param property:
        :return:
        """
        return self._properties[word][property]
    # end get

    # Remove a word from the correspondences
    def remove(self, word):
        """
        Remove a word from the correspondence
        :param word:
        :return:
        """
        del self._word2vec[word]
    # end remove

    # Clean the correspondences based on a property
    def clean(self, property, value, threshold='min', clean_punc=True):
        """
        Clean the correspondences based on a property
        :param property:
        :param value:
        :param threshold:
        :return:
        """
        # Punctuations
        puncs = [u',', u'.', u';', u':', u'!', u'?', u'(', u')', u'"', u"^", u'\'', u'``', u'...', u'-', u'*', u'\'\'',
                 u'/', u']', u'[', u'--', u'_']

        # Check each word
        for word in self._properties.keys():
            if clean_punc and word in puncs:
                self.remove(word)
            elif property in self._properties[word].keys():
                if threshold == 'min':
                    # MIN
                    if self._properties[word][property] < value:
                        self.remove(word)
                    # end if
                elif threshold == 'max':
                    # MAX
                    if self._properties[word][property] > value:
                        self.remove(word)
                    # end if
                elif threshold == 'eq':
                    # EQUAL
                    if self._properties[word][property] == value:
                        self.remove(word)
                    # end if
                elif threshold == 'neq':
                    # NOT EQUAL
                    if self._properties[word][property] != value:
                        self.remove(word)
                    # end if
                else:
                    raise Exception(u"Unknown threshold")
                # end if
            # end if
        # end for
    # end clean

    # Similar words
    def similar_words(self, vector, count=-1, measure_func='cosine'):
        """
        Similar words
        :param vector:
        :return:
        """
        # List of words with measures
        word_list = list()

        # For each word in dictionary
        for word in self._word2vec:
            # Vector
            word_vector = self[word]

            # Measure
            if measure_func == 'cosine':
                reverse = True
                measure = self._cosine_similarity(vector, word_vector)[0, 0]
            elif measure_func == 'euclidian':
                reverse = False
                measure = self._euclidian_distance(vector, word_vector)
            else:
                raise Exception(u"Unknown measure")
            # end if

            # Add to list
            word_list.append((word, measure))
        # end for

        # Sort the list by measure
        word_list.sort(key=lambda tup: tup[1], reverse=reverse)

        # Return
        if count > 0:
            return word_list[:count]
        else:
            return word_list
        # end if
    # end similar_words

    ############################################
    # Private
    ############################################

    # Euclidian distance
    def _euclidian_distance(self, vec1, vec2):
        """
        Euclidian distance
        :param vec1:
        :param vec2:
        :return:
        """
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return np.linalg.norm(vec1-vec2)
    # end euclidian_distance

    # Cosine similarity
    def _cosine_similarity(self, vec1, vec2):
        """
        Cosine similarity
        :param vec2:
        :return:
        """
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)
    # end

    @staticmethod
    def save_figure(reduced_matrix, selected_word_indexes, image, fig_size=5000):
        """
        Save figure of words
        :return:
        """
        # Figure
        plt.figure(figsize=(fig_size * 0.003, fig_size * 0.003), dpi=300)

        # Average
        mean_x = np.average(reduced_matrix[:, 0])
        mean_y = np.average(reduced_matrix[:, 1])

        # Std
        std_x = np.std(reduced_matrix[:, 0])
        std_y = np.std(reduced_matrix[:, 1])

        # Limits
        max_x = np.amax(reduced_matrix, axis=0)[0]
        max_y = np.amax(reduced_matrix, axis=0)[1]
        min_x = np.amin(reduced_matrix, axis=0)[0]
        min_y = np.amin(reduced_matrix, axis=0)[1]
        plt.xlim((mean_x - std_x, mean_x + std_x))
        plt.ylim((mean_y - std_y, mean_y + std_y))

        # Plot each words
        for word_text in selected_word_indexes.keys():
            word_index = selected_word_indexes[word_text]
            plt.scatter(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], 0.5)
            plt.text(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], word_text, fontsize=2.5)
        # end for

        # Save image
        plt.savefig(image)
        plt.close()
    # end save_figure

    ############################################
    # Override
    ############################################

    # Get vectors from word-symbols
    def __call__(self, tokens):
        """
        Get vectors from word-symbols
        :param items:
        :return:
        """
        # Resulting numpy array
        doc_array = np.array([])

        # For each token
        ok = False
        for index, word in enumerate(tokens):
            # Replace \n, \t, \r
            word_text = word.replace(u"\n", u"")
            word_text = word_text.replace(u"\t", u"")
            word_text = word_text.replace(u"\r", u"")

            # Get vector
            word_vector = self[word_text]

            # Not found
            if word_vector is None:
                word_vector = np.zeros(self.size)
            # end if

            # Stack
            if not ok:
                doc_array = word_vector
                ok = True
            else:
                doc_array = np.vstack((doc_array, word_vector))
            # end if
        # end for

        return doc_array
    # end __call__

    # Get a vector from a word-symbol
    def __getitem__(self, word):
        """
        Override get item
        :param item:
        :return:
        """
        try:
            return self._word2vec[word]
        except KeyError:
            return None
        # end if
    # end __getitem__

    ############################################
    # Static
    ############################################

    @staticmethod
    def reduction(word_embeddings, reduction='TSNE'):
        """
        Reduction
        :param word_embeddings:
        :param reduction:
        :return:
        """
        if reduction == 'TNSE':
            return Embeddings.reduction_tsne(word_embeddings)
        else:
            return Embeddings.reduction_pca(word_embeddings)
        # end if
    # end reduction

    @staticmethod
    def reduction_tsne(word_embeddings):
        """
        Reduction with TSNE
        :return:
        """
        model = TSNE(n_components=2, random_state=0)
        return model.fit_transform(word_embeddings.T)
    # end reduction_tsne

    @staticmethod
    def reduction_pca(word_embeddings):
        """
        Reduction with PCA
        :param word_embeddings:
        :return:
        """
        model = PCA(n_components=2, random_state=0)
        return model.fit_transform(word_embeddings.T)
    # end reduction_pca

# end Embeddings
