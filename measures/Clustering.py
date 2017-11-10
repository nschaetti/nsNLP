#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import numpy as np


# Clustering evaluation
class Clustering(object):
    """
    Clustering measure
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        pass
    # end __init__

    #################################
    # Static
    #################################

    # B-cubed F-1
    @staticmethod
    def bcubed_f1(clusters):
        """
        B-cubed F-1
        :param clusters:
        :return:
        """
        # F1-scores
        f_scores = list()

        # For each clusters
        for c, cluster in enumerate(clusters):
            for e, element in enumerate(cluster):
                # Precision and recall
                precision = Clustering.precision(element, cluster)
                recall = Clustering.recall(clusters, element, cluster)

                # F1
                fscore = Clustering.F1(precision, recall)

                # Add
                f_scores.append(fscore)
            # end for
        # end for

        return np.average(f_scores)
    # end bcubed_f1

    # Precision
    @staticmethod
    def precision(e, cluster):
        """
        Compute precision
        :param cluster:
        :param e:
        :return:
        """
        # Counters
        count = 0.0
        total = float(len(cluster))

        # For each element in the cluster
        for ep in cluster:
            if e.y() == ep.y():
                count += 1.0
            # end if
        # end for

        return count / total
    # end precision

    # Recall
    @staticmethod
    def recall(clusters, element, cluster):
        """
        Compute recall
        :param clusters:
        :param element:
        :param cluster:
        :return:
        """
        # Counters
        count = 0.0
        total = 0.0

        # For each clusters
        for cp in clusters:
            for ep in cp:
                # Same cluster, same class
                if cp == cluster and element.y() == ep.y():
                    count += 1.0
                    total += 1.0
                elif element.y() == ep.y():
                    total += 1.0
                # end if
            # end for
        # end for

        return count / total
    # end recall

    # F1
    @staticmethod
    def F1(precision, recall):
        """
        Precision and recall
        :param precision:
        :param recall:
        :return:
        """
        return (2.0 * precision * recall) / (precision + recall)
    # end F1

# end Clustering
