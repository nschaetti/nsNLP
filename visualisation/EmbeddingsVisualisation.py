
# Imports
import codecs
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Tools to visualize embeddings
class EmbeddingsVisualisation(object):
    """
    Tools to visualize embeddings
    """

    # Save node CSV
    @staticmethod
    def node_csv(output, nodes, captions):
        """
        Save node CSV
        :param output:
        :param nodes:
        :param captions:
        :return:
        """
        # Open the node file
        with codecs.open(output) as f:
            # Header
            f.write(u"Id,Label\n")

            # For each doc
            for document in nodes:
                document_caption = captions[document]
                f.write(u"{},{}\n".format(document, document_caption))
            # end for
        # end with
    # end node_csv

    # Save weights graph
    @staticmethod
    def weights_csv(output, elements, document2index, similarity_matrix, links=False):
        # Open the edge file
        with codecs.open(output, 'w', encoding='utf-8') as f:
            # Header
            f.write(u"Source,Target,Weight\n")

            # Compute distance between each documents
            for document1 in elements:
                for document2 in elements:
                    if document1 != document2:
                        document1_index = document2index[document1.get_path()]
                        document2_index = document2index[document2.get_path()]
                        similarity = similarity_matrix[document1_index, document2_index]
                        f.write(u"{},{},{}".format(document1_index, document2_index, similarity))
                    # end if
                # end for
            # end for
        # end with
    # end weights_csv

    # Save CSV of ordered measures
    @staticmethod
    def ordered_distances_csv(output, distances_matrix, captions, reverse=False):
        """
        Save CSV of ordered measures
        :param output: Output file
        :param distances_matrix:
        :param captions:
        :return:
        """
        # N doc
        n_docs = distances_matrix.shape[0]

        # Ordered distances
        ordered_distances = list()

        # For each distance
        for index1 in range(n_docs):
            for index2 in range(n_docs):
                if index1 != index2:
                    ordered_distances.append((index1, index2, distances_matrix[index1, index2]))
                # end if
            # end for
        # end for

        # Order distances
        ordered_distances = sorted(ordered_distances, key=lambda tup: tup[2], reverse=reverse)

        # Open file
        with codecs.open(output, 'w', encoding='utf-8') as f:
            # Header
            f.write(u"Author1,Author2,Similarity\n")
            # For each distance
            for index1, index2, distance in ordered_distances:
                f.write(u"{},{},{}\n".format(captions[index1], captions[index2], distance))
            # end for
        # end with
    # end ordered_distances_csv

    # Visualize embeddings with TSNE
    @staticmethod
    def tsne(embeddings, fig_size, output, captions=None):
        """
        Visualize embeddings with TSNE
        :return:
        """
        # Embeddings matrix
        embeddings_matrix = np.zeros((len(embeddings.keys()), embeddings[0].shape[0]))

        # Element to index
        element2index = dict()
        index2element = dict()

        # Transform to matrix
        for index, element in enumerate(embeddings.keys()):
            element2index[element] = index
            index2element[index] = element
            embeddings_matrix[index, :] = embeddings[element]
        # end for

        # Reduce with t-SNE
        model = TSNE(n_components=2, random_state=0)
        reduced_matrix = model.fit_transform(embeddings_matrix.T)

        # Show t-SNE
        plt.figure(figsize=(fig_size * 0.003, fig_size * 0.003), dpi=300)
        max_x = np.amax(reduced_matrix, axis=0)[0]
        max_y = np.amax(reduced_matrix, axis=0)[1]
        min_x = np.amin(reduced_matrix, axis=0)[0]
        min_y = np.amin(reduced_matrix, axis=0)[1]
        plt.xlim((min_x * 1.2, max_x * 1.2))
        plt.ylim((min_y * 1.2, max_y * 1.2))
        for element in embeddings:
            element_index = element2index[element]
            plt.scatter(reduced_matrix[element_index, 0], reduced_matrix[element_index, 1], 0.5)
            if captions is not None:
                element_caption = captions[element]
                plt.text(reduced_matrix[element_index, 0], reduced_matrix[element_index], element_caption, fontsize=2.5)
            else:
                plt.text(reduced_matrix[element_index, 0], reduced_matrix[element_index], str(element), fontsize=2.5)
            # end if
        # end for

        # Save image
        plt.savefig(output)
    # end tsne

# end EmbeddingsVisualisation
