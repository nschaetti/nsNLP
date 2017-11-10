
# Imports
import numpy as np
import spacy
import Oger
import mdp
import spacy
import scipy.sparse
import nodes
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
                 w_sparsity=0.1, w_in=None, w=None, model='word2echo', state_gram=1, direction='both',
                 word_embeddings=None, gamma=1, n_threads=1):
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
        :param word_embeddings:
        :param gamma: Parameter for Echo2Word model
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
        self._state_gram = int(state_gram)
        self._direction = direction
        self._word_embeddings = word_embeddings
        self._gamma = gamma
        self._n_threads = n_threads

        # Direction
        if direction == 'both':
            self._side_size = 2
        else:
            self._side_size = 1
        # end if

        # Input reservoir dimension
        if self._word_embeddings is None:
            self._input_dim = converter.get_n_inputs()
        else:
            self._input_dim = self._word_embeddings.size
        # end if

        # Set of example
        self._examples = list()

        # Create the reservoir
        self._wordecho = nodes.WordEchoNode(input_dim=self._input_dim, output_dim=self._output_dim,
                                                    input_scaling=input_scaling, leak_rate=leaky_rate,
                                                    spectral_radius=spectral_radius,
                                                    sparsity=input_sparsity, w_sparsity=w_sparsity,
                                                    w_in=w_in, use_sparse_matrix=True, direction=direction)

        # Components
        self._readout = None
        self._flow = None
        self._scheduler = None
        self._context = None

        # Reset state at each call
        self._wordecho.reset_states = True

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
            # Compute embeddings size
            embedding_side_size = 2 if self._direction == 'both' else 1

            # Embedding size
            if self._model == 'echo2word':
                emb_size = self._output_dim*self._state_gram*embedding_side_size+1
            elif self._model == 'word2echo':
                emb_size = (self._output_dim*self._state_gram*embedding_side_size+1)*self._gamma*2
            else:
                raise Exception(u"Unknown model {}".format(self._model))
            # end if

            # New embeddings
            emb = Embeddings(size=emb_size)

            # For Word2Echo
            if self._model == 'echo2word':
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
            elif self._model == 'word2echo':
                print(self._readout.beta.shape)
                # Add each word with vectors and count
                for word in self._converter.words():
                    # Word index
                    word_index = self._converter.get_word_index(word)

                    # Vector for this word
                    word_vector = np.zeros(emb_size)

                    # For each position
                    j = 0
                    for i in np.arange(word_index, self._readout.beta.shape[1], self._converter.voc_size):
                        word_vector[j:j+self._readout.beta.shape[0]] = self._readout.beta[:, i]
                        j += self._readout.beta.shape[0]
                    # end for

                    # Add to embeddings
                    emb.add(word, word_vector)

                    # Set count
                    emb.set(word, 'count', self._converter.get_word_count(word))
                # end if
            # end if

            return emb
        else:
            return None
        # end if
    # end export_embeddings

    # Add an example
    def add(self, x):
        """
        Add text example
        :param x: List of vector representations
        """
        # Input through converter
        converter_inputs = self._converter(x)

        # Add input outputs
        if self._word_embeddings is not None:
            self._examples.append((self._word_embeddings(x), converter_inputs))
        else:
            self._examples.append((converter_inputs, converter_inputs))
        # end if
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
        for (x, y) in self._examples:
            # Add to data
            X.append(x)

            # Generate output
            if self._model == 'echo2word':
                Y.append(y)
            else:
                Y.append(self._word2echo_outputs(y))
            # end if
        # end for

        # Extract giving the model
        if self._model == 'word2echo':
            data = Word2Echo.data_word2echo(X, Y)
        elif self._model == 'echo2word':
            data = Word2Echo.data_echo2word(X, Y)
        # end if

        # Train the model
        self._flow.train(data)

        # Trained
        self._trained = True
    # end train

    # Reset learning but keep reservoir
    def reset_model(self):
        """
        Reset learning but keep reservoir
        :return:
        """
        del self._readout, self._flow

        # Reset dataset
        self._examples = list()

        # Ridge Regression
        self._readout = Oger.nodes.RidgeRegressionNode()

        # Context states
        if self._model == 'echo2word':
            self._context = nodes.ContextStateNode(direction=self._direction,
                                                   input_size=self._output_dim * self._side_size,
                                                   state_gram=self._state_gram)
        # end if

        # Flow
        if self._model == 'echo2word':
            self._flow = mdp.Flow([self._wordecho, self._context, self._readout], verbose=0)
        else:
            self._flow = mdp.Flow([self._wordecho, self._readout], verbose=0)
        # end if
    # end reset_model

    ###############################################
    # Private
    ###############################################

    # Get Echo2Word outputs
    def _word2echo_outputs(self, x):
        """
        Get Echo2Word outputs
        :param x:
        :return:
        """
        # Side outputs size
        side_size = int(self._input_dim * self._gamma)

        # Output matrix
        y = scipy.sparse.csr_matrix((x.shape[0], side_size * 2))

        # Zero inputs
        zero_inputs = scipy.sparse.csr_matrix((1, side_size))

        # For each token in inputs
        for i in np.arange(0, x.shape[0]):
            # Current inputs
            current_inputs = scipy.sparse.csr_matrix((1, side_size * 2))

            # Before inputs
            if i == 0:
                current_inputs[0, :side_size] = zero_inputs
            else:
                # Start, end indexes
                start = i - self._gamma
                end = i

                # Limits
                if start < 0:
                    start = 0
                # end if

                # Temporary inputs
                tmp_inputs = x[start:end, :].tolil()
                tmp_inputs.reshape((1, tmp_inputs.shape[0]*tmp_inputs.shape[1]))

                # Set
                current_inputs[0, side_size - tmp_inputs.shape[1]:side_size] = tmp_inputs
            # end if

            # After inputs
            if i == x.shape[0] - 1:
                current_inputs[0, side_size:] = zero_inputs
            else:
                # Start, end indexes
                start = i+1
                end = i+1+self._gamma

                # Limits
                if end > x.shape[0]:
                    end = x.shape[0]
                # end if

                # Temporary inputs
                tmp_inputs = x[start:end, :].tolil()
                tmp_inputs.reshape((1, tmp_inputs.shape[0]*tmp_inputs.shape[1]))

                # Get
                current_inputs[0, side_size:side_size+tmp_inputs.shape[1]] = tmp_inputs
            # end if

            # Set in y
            y[i, :] = current_inputs
        # end for

        return y
    # end _echo2word_outputs

    ###############################################
    # Static
    ###############################################

    # Create training data for word2echo
    @staticmethod
    def data_word2echo(X, Y):
        """
        Create training data for word2echo
        :param X:
        :param Y:
        :return:
        """
        # Data
        return [None, zip(X, Y)]
    # end _extract_echo2word

    # Create training data for echo2word
    @staticmethod
    def data_echo2word(X, Y):
        """
        Create training data for word2echo
        :param X:
        :param Y:
        :return:
        """
        # Data
        return [None, None, zip(X, Y)]
    # end data_echo2word

    # Create a Word2Echo model
    @staticmethod
    def create(rc_size, rc_spectral_radius, rc_leak_rate, rc_input_scaling, rc_input_sparsity,
               rc_w_sparsity, model_type, direction, w=None, voc_size=10000, uppercase=False, state_gram=1,
               converter=None, word_embeddings=None, gamma=1, n_threads=2):
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
        if converter is None:
            converter = OneHotConverter(voc_size=voc_size, uppercase=uppercase)
        # end if

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
            state_gram=state_gram,
            word_embeddings=word_embeddings,
            gamma=gamma,
            n_threads=n_threads
        )

        return word2echo_model
    # end create

# end Word2Echo
