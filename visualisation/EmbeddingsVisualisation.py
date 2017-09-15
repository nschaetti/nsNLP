
# Imports
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Tools to visualize embeddings
class EmbeddingsVisualisation(object):
    """
    Tools to visualize embeddings
    """

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
