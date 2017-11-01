
# Imports
import numpy as np
import spacy
import Oger
import mdp
import spacy
import scipy.sparse
from converters.OneHotConverter import OneHotConverter
from nsNLP.embeddings.Embeddings import Embeddings

###########################################################
# Class
###########################################################


# Echo State Network to predict the next word
class Word2Echo(object):
    """
    Echo State Network to predict the next word
    """

    # Constructor
    def __init__(self, converter, size, leaky_rate, spectral_radius, input_scaling=0.25, input_sparsity=0.1,
                 w_sparsity=0.1, w_in=None, w=None, model='word2echo', state_gram=1, direction='both'):
        """
        Constructor
        :param converter: Text converter
        :param size:
        :param leaky_rate:
        :param spectral_radius:
        :param input_scaling:
        :param input_sparsity:
        :param w_sparsity:
        :param w:
        :param model: word2echo predict the next word, echo2word predict word from surounding states.
        :param direction: Direction of the prediction
        """
        # Properties
        self._converter = converter
        self._embeddings_size = size
        self._output_dim = size
        self._leaky_rate = leaky_rate
        self._spectral_radius = spectral_radius
        self._input_scaling = input_scaling
        self._input_sparsity = input_sparsity
        self._w_sparsity = w_sparsity
        self._w_in = w_in
        self._w = w
        self._model = model
        self._trained = False
        self._state_gram = state_gram
        self._direction = direction

        # Input reservoir dimension
        self._input_dim = converter.get_n_inputs()

        # Set of example
        self._examples = list()

        # Create the reservoir
        self._reservoir = Oger.nodes.LeakyReservoirNode(input_dim=self._input_dim, output_dim=self._output_dim,
                                                        input_scaling=input_scaling, leak_rate=leaky_rate,
                                                        spectral_radius=spectral_radius,
                                                        sparsity=input_sparsity, w_sparsity=w_sparsity,
                                                        w_in=w_in, use_sparse_matrix=True)

        # Components
        self._readout = None
        self._join = None
        self._flow = None

        # Reset state at each call
        self._reservoir.reset_states = True

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Init components
        self.reset_model()
    # end __init__

    ###############################################
    # Properties
    ###############################################

    # Total tokens in the dataset
    @property
    def token_count(self):
        """
        Total tokens in the dataset
        :return:
        """
        return self._converter.token_count
    # end token_count

    # Vocabulary size of the model
    @property
    def voc_size(self):
        return self._converter.voc_size
    # end voc_size

    ###############################################
    # Public
    ###############################################

    # Export word embeddings
    def export_embeddings(self):
        """
        Export word embeddings
        """
        if self._trained:
            # New embeddings
            emb = Embeddings()

            # Add each word with vectors and count
            for word in self._converter.words():
                # Word index
                word_index = self._converter.get_word_index(word)

                # Get vector in Wout
                word_vector = self._readout.beta[:, word_index]

                # Add
                emb.add(word, word_vector)

                # Set count
                emb.set(word, 'count', self._converter.get_word_count(word))
            # end for

            return emb
        else:
            return None
        # end if
    # end export_embeddings

    # Import word embeddings
    def import_embeddings(self, embeddings):
        """
        Import embeddings
        :param embeddings:
        :return:
        """
        pass
    # end import_embeddings

    # Add an example
    def add(self, x):
        """
        Add text example
        :param x: List of vector representations
        """
        self._examples.append(self._converter(x))
    # end add

    # Extract embeddings
    def extract(self):
        """
        Extract embeddings
        """
        # Input and output
        X = list()
        Y = list()

        # For each example
        for x in self._examples:
            # Add to data
            X.append(x)
            Y.append(x)
        # end for

        # List of states
        joined_states_lr = list()
        joined_states_rl = list()

        # Compute the states from left to right
        if self._direction == 'both' or self._direction == 'lr':
            for x in X:
                tmp_states = self._reservoir.execute(x)
                joined_states_lr.append(self._join.execute(tmp_states)[:-1, :])
            # end for
        # end if

        # Compute the states from right to left
        if self._direction == 'both' or self._direction == 'rl':
            for x in X:
                reversed_inputs = scipy.sparse.csr_matrix(np.flip(x.toarray(), axis=0))
                tmp_states = self._reservoir.execute(reversed_inputs)
                joined_states_rl.append(np.flip(self._join.execute(tmp_states)[:-1, :], axis=0))
            # end for
        # end if

        # Merge both direction if needed
        if self._direction == 'both':
            merge_states = list()
            for index, state_lr in enumerate(joined_states_lr):
                """print(state_lr.shape)
                print(state_lr[0:2, :])
                print(joined_states_rl[index].shape)
                print(joined_states_rl[index][0:2, :])"""
                merge_states.append(np.hstack((state_lr, joined_states_rl[index])))
            # end for
        elif self._direction == 'lr':
            merge_states = joined_states_lr[:-1]
        elif self._direction == 'rl':
            merge_states = joined_states_rl.reverse()[1:]
        # end if

        # Data
        data = [zip(merge_states, Y)]

        # Train the model
        self._flow.train(data)

        # Trained
        self._trained = True
    # end train

    # Get word embeddings
    """def get_word_embeddings(self):
        ""
        Get word embeddings
        :return:
        ""
        if self._trained:
            if self._n_word_embeddings == 1:
                return self._readout.beta
            else:
                n_words = self._readout.beta.shape[1]
                single_dimension = self._readout.beta.shape[0]
                word_embeddings = np.zeros((single_dimension * self._n_word_embeddings, n_words))
                # For each word embeddings
                for word_index in range(n_words):
                    # For each word embedding
                    for embedding_index in range(self._n_word_embeddings):
                        word_embeddings[embedding_index*single_dimension, word_index] = self._readout.beta[:, embedding_index*n_words]
                # end for
            # end if
        else:
            raise ReservoirNotTrainedException(u"Reservoir not trained!")
        # end if
    # end get_word_embeddings"""

    # Reset learning but keep reservoir
    def reset_model(self):
        """
        Reset learning but keep reservoir
        :return:
        """
        del self._readout, self._flow, self._join

        # Reset dataset
        self._examples = list()

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Join
        self._join = Oger.nodes.JoinedStatesNode(input_dim=self._output_dim, joined_size=self._state_gram,
                                                 fill_before=True)

        # Flow
        self._flow = mdp.Flow([self._readout], verbose=0)
    # end reset_model

    ###############################################
    # Private
    ###############################################

    # Generate outputs
    def _generate_dataset(self, x):
        """
        Generate outputs
        :param x:
        :return:
        """
        return 1, 2
    # end _generate_outputs

    ###############################################
    # Static
    ###############################################

    # Create a Word2Echo model
    @staticmethod
    def create(rc_size, rc_spectral_radius, rc_leak_rate, rc_input_scaling, rc_input_sparsity,
               rc_w_sparsity, model_type, direction, w=None, voc_size=10000, uppercase=False, state_gram=1):
        """
        Create a Word2Echo model
        :param rc_size:
        :param rc_spectral_radius:
        :param rc_leak_rate:
        :param rc_input_scaling:
        :param rc_input_sparsity:
        :param rc_w_sparsity:
        :param w:
        :param voc_size:
        :param uppercase:
        :return:
        """
        # Converter
        converter = OneHotConverter(voc_size=voc_size, uppercase=uppercase)

        # Create the Word2Echo
        word2echo_model = Word2Echo \
        (
            size=rc_size,
            input_scaling=rc_input_scaling,
            leaky_rate=rc_leak_rate,
            input_sparsity=rc_input_sparsity,
            converter=converter,
            spectral_radius=rc_spectral_radius,
            w_sparsity=rc_w_sparsity,
            w=w,
            model=model_type,
            direction=direction,
            state_gram=state_gram
        )

        return word2echo_model
    # end create

# end Word2Echo
