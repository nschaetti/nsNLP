
# Imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Distance measures tools
class DistanceMeasures(object):
    """
    Distance measures tools
    """

    # Similarity matrix
    @staticmethod
    def similarity_matrix(doc2vec, measure='cosine'):
        """
        Similarity matrix
        :param doc2vec:
        :param measure:
        :return:
        """
        # Number of docs
        n_docs = len(doc2vec.keys())

        # Similarity matrix
        similarity_matrix = np.zeros((n_docs, n_docs))

        # Compute similarity matrix
        for element1 in doc2vec.keys():
            for element2 in doc2vec.keys():
                if element1 != element2:
                    distance = cosine_similarity(doc2vec[element1], doc2vec[element2])
                    similarity_matrix[doc2vec[element1], doc2vec[element2]] = distance
                # end if
            # end for
        # end for

        return similarity_matrix
    # end similarity_matrix

    # Link matrix
    @staticmethod
    def link_matrix(similarity_matrix, sigma=1.65):
        """
        Link matrix
        :param similarity_matrix:
        :param sigma:
        :return:
        """
        # Number of docs
        n_docs = similarity_matrix.shape[0]

        # Links matrix
        links_matrix = np.zeros((n_docs, n_docs))

        # Compute links matrix
        for index in range(n_docs):
            # Get the row
            document_row = similarity_matrix[index, :]

            # Remove self relation
            document_row_cleaned = np.delete(document_row, index)

            # Threshold
            average_similarity = np.average(document_row_cleaned)
            distance_threshold = sigma * np.std(document_row_cleaned)

            # Make
            links_matrix[index, document_row - average_similarity >= distance_threshold] = 1.0
        # end for

        return links_matrix
    # end link_matrix

    # Cosine similarity
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Cosine similarity
        :param vec1:
        :param vec2:
        :return:
        """
        return cosine_similarity(vec1, vec2)
    # end cosine_similarity

# end DistanceMeasures
