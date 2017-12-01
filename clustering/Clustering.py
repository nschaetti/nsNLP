#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.cluster
import networkx as nx
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


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
        self._samples = list()
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

        # Distance again
        self._distance_measured = False
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

        # Distance again
        self._distance_measured = False
    # end update

    # Compute hierarchical clustering
    def hierarchical_clustering(self, dendogram_file=None):
        """
        Compute hierarchical clustering
        :param dendogram: Output filename for the dendogram
        :return:
        """
        # Array of vectors
        X = np.array([])

        # Add each sample vector
        for index, sample in enumerate(self._samples):
            if index == 0:
                X = self._sample_vectors[sample]
            else:
                X = np.vstack((X, self._sample_vectors[sample]))
            # end if
        # end for

        # Clustering
        Z = linkage(X, 'ward')

        # Dendogram
        if dendogram_file:
            plt.figure()
            plt.title(u"Hierarchical Clustering Dendogram")
            plt.xlabel('sample index')
            plt.ylabel('distance')
            dendrogram(
                Z,
                leaf_rotation=90.,
                leaf_font_size=8,
            )
            plt.savefig(dendogram_file)
        # end if

        return Z
    # end hierarchical_clustering

    # Compute clusters with K-Means
    def k_means(self, k, random_state=0):
        """
        Compute clusters with K-Means
        :param k:
        :return:
        """
        # Array of vectors
        X = np.array([])

        # Create clusters
        clusters = [list() for i in range(k)]

        # Add each sample vector
        for index, sample in enumerate(self._samples):
            if index == 0:
                X = self._sample_vectors[sample]
            else:
                X = np.vstack((X, self._sample_vectors[sample]))
            # end if
        # end for

        # Do K-means clustering
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=random_state).fit(X)

        # For each labels
        for i in range(kmeans.labels_.shape[0]):
            # Label
            label = kmeans.labels_[i]

            # Sample
            sample = self._samples[i]

            # Add sample
            clusters[label].append(sample)
        # end for

        return clusters
    # end cluster

    # Compute clusters with adaptive threshold
    def adaptive_threshold(self, measure='cosine'):
        """
        Compute clusters with adaptive threshold
        :param measure:
        :return:
        """
        # Compute distance if necessary
        self._compute_distance_matrix(measure=measure)

        # List of edges
        edges = list()

        # Average and std
        avg = np.average(self._distance_matrix)
        std = np.std(self._distance_matrix)

        # Threshold
        threshold = avg + std*2.0

        # For each tuple of samples
        for i, s1 in enumerate(self._samples):
            for j, s2 in enumerate(self._samples):
                if s1 != s2:
                    # Pass threshold
                    if measure == 'cosine' and self._distance_matrix[i, j] > threshold:
                        edges.append((i, j))
                    elif self._distance_matrix[i, j] < threshold:
                        edges.append((i, j))
                    # end if
                # end if
            # end for
        # end for

        # New graph
        G = nx.Graph()

        # Add edges
        G.add_edges_from(edges)

        # Set of connected components
        connected_components = list(nx.connected_components(G))

        # Clusters
        clusters = [list() for i in range(len(connected_components))]

        # Add sample to each cluster
        for index, c in enumerate(connected_components):
            for e in c:
                # Sample
                sample = self._samples[e]

                # Add
                self._clusters[index].append(sample)
            # end for
        # end for

        return clusters
    # end adaptive_threshold

    # Get distance list
    def get_distance_list(self, measure='cosine', sort=False):
        """
        Get distance list
        :param measure:
        :param sort:
        :return:
        """
        # Compute distance matrix, if necessary
        if not self._distance_measured:
            self._compute_distance_matrix(measure=measure)
        # end if

        # List of distance
        distance_list = list()

        # For each combination of sample
        for i in range(len(self._samples)):
            for j in range(len(self._samples)):
                if i != j:
                    # Samples
                    s1 = self._samples[i]
                    s2 = self._samples[j]

                    # Only samples
                    sample_list = ((e[0], e[1]) for e in distance_list)

                    # Not already in list
                    if (s1, s2) and (s2, s1) not in sample_list:
                        distance_list.append((s1, s2, self._distance_matrix[i, j]))
                    # end if
                # end if
            # end for
        # end for

        # Sort?
        if sort:
            distance_list = distance_list.sort(key=lambda x: x[2])
        # end if

        return distance_list
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
