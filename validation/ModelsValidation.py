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
import sys
import numpy as np
from scipy import stats
import nsNLP.tools as tools
from .CrossValidation import CrossValidation


# Models validation
class ModelsValidation(object):
    """
    Models validation
    """

    # Constructor
    def __init__(self, dataset, k=10, shuffle=True):
        """
        Constructo
        :param dataset:
        :param shuffle:
        """
        # Cross validation
        self._k = k
        self._cross_validation = CrossValidation(dataset=dataset, k=k, shuffle=shuffle)

        # List of models to make prediction
        self._models = list()
    # end __init__

    #######################################
    # Public
    #######################################

    # Add
    def add_model(self, model):
        """
        Add a model
        :param model:
        :return:
        """
        self._models.append(model)
    # end add_model

    # Compare models
    def compare(self):
        """
        Compare models
        :return:
        """
        # Results
        results = dict()
        comparisons = dict()

        # For each model
        for model in self._models:
            results[model.name()] = self._evaluate(model)
        # end for

        # Compare each models
        for model1 in self._models:
            for model2 in self._models:
                if model1 != model2:
                    comparisons[(model1.name(), model2.name())] = stats.ttest_ind(results[model1.name()],
                                                                                  results[model2.name()])[1]
                # end if
            # end for
        # end for

        return results, comparisons
    # end compare

    #######################################
    # Private
    #######################################

    # Evaluate model
    def _evaluate(self, model):
        """
        Evaluate model
        :param model:
        :return:
        """
        # Results
        results = np.zeros(self._k)

        # Log
        sys.stdout.write(u"Evaluating model {}\n".format(model.name()))

        # For each fold
        k = 0
        for training_set, test_set in self._cross_validation:
            # Training
            for sample in training_set:
                model.train(sample.get_text(), sample.get_author().get_name(), verbose=True)
            # end for

            # Success rate on test set
            results[k] = tools.Metrics.success_rate(model, test_set)
            sys.stdout.write(u"Test success rate for {}, fold {} : {}\n".format(model.name(), k, results[k]))

            # Next fold
            k += 1
        # end for

        # Average
        sys.stdout.write(u"Average test success rate for {} : {}\n".format(model.name(), np.average(results)))

        return results
    # end _evaluate

# end KFoldCrossValidation
