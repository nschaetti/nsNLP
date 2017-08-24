#
# nsNLP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import random
import copy


# Two classes cross validation
class TwoClassesCrossValidation(object):
    """
    Two classes cross validation
    """

    # Constructor
    def __init__(self, positive_dataset=list(), negative_dataset=list(), k=10, shuffle=True):
        """
        Constructo
        :param dataset:
        :param shuffle:
        """
        # Dataset
        self._positives = positive_dataset
        self._negatives = negative_dataset

        # Sizes
        self._k = k
        self._n_positive_samples = len(self._positives)
        self._n_negative_samples = len(self._negatives)

        # Fold's sizes
        self._positive_fold_size = int(self._n_positive_samples / self._k)
        self._negative_fold_size = int(self._n_negative_samples / self._k)

        # Shuffle
        self._shuffle = shuffle
        if shuffle:
            random.shuffle(self._positives)
            random.shuffle(self._negatives)
        # end if

        # Position
        self._fold_pos = 0
    # end __init__

    #######################################
    # Override
    #######################################

    # Add positive sample
    def add_positive(self, sample):
        """
        Add positive sample
        :param sample:
        :return:
        """
        self._positives.append(sample)
    # end add_positive

    # Add negative sample
    def add_negative(self, sample):
        """
        Add negative sample
        :param sample:
        :return:
        """
        self._negatives.append(sample)
    # end add_negative

    #######################################
    # Override
    #######################################

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next
    def next(self):
        """
        Next element
        :return:
        """
        if self._fold_pos == 0:
            # Sizes
            self._n_positive_samples = len(self._positives)
            self._n_negative_samples = len(self._negatives)

            # Fold's sizes
            self._positive_fold_size = int(self._n_positive_samples / self._k)
            self._negative_fold_size = int(self._n_negative_samples / self._k)

            # Shuffle
            if self._shuffle:
                random.shuffle(self._positives)
                random.shuffle(self._negatives)
            # end if
        # end if

        if self._fold_pos < self._k:
            # Total
            positive_train_set = copy.copy(self._positives)
            negative_train_set = copy.copy(self._negatives)

            # Test set
            positive_test_set = positive_train_set[self._positive_fold_size * self._fold_pos:self._positive_fold_size * self._fold_pos + self._positive_fold_size]
            negative_test_set = negative_train_set[self._negative_fold_size * self._fold_pos:self._negative_fold_size * self._fold_pos + self._negative_fold_size]

            # Final positive training set
            for sample in positive_test_set:
                positive_train_set.remove(sample)
            # end for

            # Final negative training set
            for sample in negative_test_set:
                negative_train_set.remove(sample)
            # end for

            # Next fold
            self._fold_pos += 1

            # Return sets
            return positive_train_set, negative_train_set, positive_test_set, negative_test_set
        else:
            self._fold_pos = 0
            raise StopIteration()
        # end if
    # end next

# end TwoClassesCrossValidation
