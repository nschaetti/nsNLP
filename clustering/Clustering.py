#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Clustering tool
class Clutering(object):
    """
    Clustering
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Variables
        self._samples= list()
        self._sample_vectors = dict()
        self._distance_matrix = np.array([])
        self._text2index = dict()
        self._index2text = dict()
        self._n_texts = 0
        self._distance_measured = False
        self._clusters = list()
    # end __init__

    ########################################
    # Public
    ########################################

    # Add a sample
    def add(self, sample, vector):
        """
        Add a sample
        :param name:
        :param vector:
        :return:
        """
        # Check
        if sample in self._samples:
            raise Exception(u"Sample already in the clustering")
        # end if

        # Add name and vector
        self._samples.append(sample)
        self._sample_vectors[sample] = vector

        # Set dict
        self._text2index[sample] = self._n_texts
        self._index2text[self._n_texts] = sample

        # Inc
        self._n_texts += 1
    # end add

    # Update a sample
    def update(self, sample, vector):
        """
        Update a sample
        :param name:
        :param vector:
        :return:
        """
        # Check
        if sample not in self._samples:
            raise Exception(u"Sample not in the clustering")
        # end if

        # Update vector
        self._sample_vectors[sample] = vector
    # end update

    # Clusters
    def clusters(self, measure='cosine'):
        """
        Cluster
        :param measure:
        :return:
        """
        # Compute distance matrix
        self._compute_distance_matrix(measure=measure)
    # end cluster

    # Get distance list
    def get_distance_list(self, sorted=False):
        """
        Get distance list
        :param sorted:
        :return:
        """
        # List of distance
        pass
    # end

    ########################################
    # Private
    ########################################

    # Compute distance matrix
    def _compute_distance_matrix(self, measure='cosine', force=False):
        """
        Compute distance matrix
        :return:
        """
        if not self._distance_measured or force:
            # Create matrix
            self._distance_matrix = np.zeros((self._n_texts, self._n_texts))

            # For each text
            for i, sample1 in self._samples:
                for j, sample2 in self._samples:
                    # Not the same
                    if i != j:
                        # Both index
                        index1 = self._text2index[sample1]
                        index2 = self._text2index[sample2]

                        # Not done yet
                        if self._distance_matrix[index1, index2] == 0:
                            # Vectors
                            vector1 = self._sample_vectors[sample1]
                            vector2 = self._sample_vectors[sample2]

                            # Compute distance
                            if measure == 'cosine':
                                distance = self._cosine_similarity(vec1=vector1, vec2=vector2)
                            elif measure == 'euclidian':
                                distance = self._euclidian_distance(vec1=vector1, vec2=vector2)
                            elif measure == 'manhatan':
                                pass
                            # end if

                            # Set distance
                            self._distance_matrix[index1, index2] = distance
                            self._distance_matrix[index2, index1] = distance
                        # end if
                    # end if
                # end for
            # end for

            # Distance measured
            self._distance_measured = True
        # end if
    # end _compute_distance_matrix

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
        return np.linalg.norm(vec1 - vec2)

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

# end Clutering
