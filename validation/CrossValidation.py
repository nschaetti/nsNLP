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


# Cross validation
class CrossValidation(object):
    """
    Cross validation
    """

    # Constructor
    def __init__(self, dataset, k=10, shuffle=True):
        """
        Constructo
        :param dataset:
        :param shuffle:
        """
        self._dataset = dataset
        self._n_samples = len(dataset)
        self._k = k
        self._folds = list()
        self._fold_size = int(self._n_samples / k)
        if shuffle:
            random.shuffle(self._dataset)
        # end if
        self._fold_pos = 0
    # end __init__

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
        if self._fold_pos < self._k:
            # Total
            train_set = copy.copy(self._dataset)

            # Test set
            test_set = train_set[self._fold_size*self._fold_pos:self._fold_size*self._fold_pos+self._fold_size]

            # Final training set
            for sample in test_set:
                train_set.remove(sample)
            # end for

            # Next fold
            self._fold_pos += 1

            # Return sets
            return train_set, test_set
        else:
            self._fold_pos = 0
            raise StopIteration()
        # end if
    # end next

# end KFoldCrossValidation
