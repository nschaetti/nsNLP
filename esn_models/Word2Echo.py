
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
                 w_sparsity=0.1, w_in=None, w=None, model='word2echo', state_gram=1, direction='both',
                 word_embeddings=None, gamma=1):
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
        self._state_gram = state_gram
        self._direction = direction
        self._word_embeddings = word_embeddings
        self._gamma = gamma

        # Input reservoir dimension
        if self._word_embeddings is None:
            self._input_dim = converter.get_n_inputs()
        else:
            self._input_dim = self._word_embeddings.size
        # end if

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
            # Compute embeddings size
            embedding_side_size = 2 if self._direction == 'both' else 1

            # Embedding size
            if self._model == 'word2echo':
                emb_size = self._output_dim*self._state_gram*embedding_side_size+1
            elif self._model == 'echo2word':
                emb_size = self._output_dim*self._state_gram**embedding_side_size*(self._gamma*embedding_side_size+1)
            else:
                raise Exception(u"Unknown model {}".format(self._model))
            # end if

            # New embeddings
            emb = Embeddings(size=emb_size)

            # For Word2Echo
            if self._model == 'word2echo':
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
            elif self._model == 'echo2word':
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
        # Extract giving the model
        if self._model == 'word2echo':
            self._extract_word2echo()
        elif self._model == 'echo2word':
            self._extract_echo2word()
        # end if
    # end train

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

    # Get Echo2Word outputs
    def _echo2word_outputs(self, x):
        """
        Get Echo2Word outputs
        :param x:
        :return:
        """
        # Side outputs size
        side_size = int(self._output_dim * self._gamma)

        # Output matrix
        y = scipy.sparse.csr_matrix((x.shape[0], side_size * 2))

        # Zero inputs
        zero_inputs = scipy.sparse.csr_matrix(side_size)

        # For each token in inputs
        for i in np.arange(0, x.shape[0]):
            # Current inputs
            current_inputs = scipy.sparse.csr_matrix(side_size * 2)

            # Before inputs
            if i == 0:
                current_inputs[:side_size] = zero_inputs
            else:
                # Start, end indexes
                start = i - self._gamma
                end = i

                # Limits
                if start < 0:
                    start = 0
                # end if

                # Temporary inputs
                tmp_inputs = y[start:end, :].flatten()

                # Set
                current_inputs[side_size - tmp_inputs.shape[0]:side_size] = tmp_inputs
            # end if

            # After inputs
            if i == x.shape[0] - 1:
                current_inputs[side_size:] = zero_inputs
            else:
                # Start, end indexes
                start = i+1
                end = i+1+self._gamma

                # Limits
                if end > y.shape[0]:
                    end = y.shape[0] + 1
                # end if

                # Temporary inputs
                tmp_inputs = y[start:end, :].flatten()

                # Get
                current_inputs[side_size:side_size+tmp_inputs.shape[0]] = y[start:end, :]
            # end if

            # Set in y
            y[i, :] = current_inputs
        # end for

        return y
    # end _echo2word_outputs

    # Extract with echo2word
    def _extract_echo2word(self):
        """
        Extract with echo2word
        :return:
        """
        # Input and output
        X = list()
        Y = list()

        # For each example
        for (x, y) in self._examples:
            # Add to data
            X.append(x)
            Y.append(self._echo2word_outputs(y))
        # end for

        # List of states
        joined_states_lr = list()
        joined_states_rl = list()

        # Compute the states from left to right
        if self._direction == 'both' or self._direction == 'lr':
            for x in X:
                tmp_states = self._reservoir.execute(x)
                joined_states_lr.append(self._join.execute(tmp_states)[1:, :])
            # end for
        # end if

        # Compute the states from right to left
        if self._direction == 'both' or self._direction == 'rl':
            for x in X:
                reversed_inputs = self._flip_matrix(x)
                tmp_states = self._reservoir.execute(reversed_inputs)
                joined_states_rl.append(np.flip(self._join.execute(tmp_states)[1:, :], axis=0))
            # end for
        # end if

        # Merge both direction if needed
        if self._direction == 'both':
            merge_states = list()
            for index, state_lr in enumerate(joined_states_lr):
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
    # end _extract_echo2word

    # Extract with word2echo
    def _extract_word2echo(self):
        """
        Extract with word2echo
        :return:
        """
        # Input and output
        X = list()
        Y = list()

        # For each example
        for (x, y) in self._examples:
            # Add to data
            X.append(x)
            Y.append(y)
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
                reversed_inputs = self._flip_matrix(x)
                tmp_states = self._reservoir.execute(reversed_inputs)
                joined_states_rl.append(np.flip(self._join.execute(tmp_states)[:-1, :], axis=0))
                # end for
        # end if

        # Merge both direction if needed
        if self._direction == 'both':
            merge_states = list()
            for index, state_lr in enumerate(joined_states_lr):
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
    # end _extract_word2echo

    # Flip sparse matrix
    def _flip_matrix(self, m):
        """
        Flip sparse matrix
        :param m:
        :return:
        """
        # New CSR
        m_flip = scipy.sparse.csr_matrix((m.shape[0], m.shape[1]))

        # Go backward
        j = 0
        for i in np.arange(m.shape[0]-1, -1, -1):
            m_flip[j, :] = m[i, :]
            j += 1
        # end for

        return m_flip
    # end _flip_matrix

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
               rc_w_sparsity, model_type, direction, w=None, voc_size=10000, uppercase=False, state_gram=1,
               converter=None, word_embeddings=None, gamma=1):
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
        #else:
        #     converter.reset_word_count()
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
            gamma=gamma
        )

        return word2echo_model
    # end create

# end Word2Echo
