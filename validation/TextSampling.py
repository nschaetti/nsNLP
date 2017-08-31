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


# Text Sampling
class TextSampling(object):
    """
    TextSampling
    """

    # Constructor
    def __init__(self, text, n_min_lines, n_max_lines):
        """
        Constructor
        """
        # Properties
        self._text = text
        self._n_min_lines = n_min_lines
        self._n_max_lines = n_max_lines
        self._lines = []
        self._n_lines = 0

        # Clean text
        self._clean_text()
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Add text
    def add_text(self, text):
        """
        Add text
        :param text:
        :return:
        """
        self._text += u"\n" + text
        self._clean_text()
    # end add_text

    # Add texts
    def add_texts(self, texts):
        """
        Add texts
        :param texts:
        :return:
        """
        for text in texts:
            self.add_text(text)
        # end for
    # end add_texts

    # Get number of lines
    def n_lines(self):
        """
        Get number of lines
        :return:
        """
        return self._n_lines
    # end n_lines

    ##############################################
    # Override
    ##############################################

    # Call
    def __call__(self):
        """
        Call
        :return:
        """
        samples_text = u""

        # Random number of lines
        n_random_lines = random.randint(self._n_min_lines, self._n_max_lines)

        # Select lines
        for j in range(n_random_lines):
            # Select line
            n_random_pos = random.randint(0, self._n_lines)

            # Add
            samples_text += samples_text[n_random_pos] + u"\n"
        # end for

        return samples_text
    # end __call__

    ##############################################
    # Private
    ##############################################

    # Clean text
    def _clean_text(self):
        """
        Clean text
        :return:
        """
        # Remove blank line
        for i in range(20):
            self._text = self._text.replace(u"\n\n", u"\n")
        # end for

        # Get all lines
        self._lines = self._text.split(u"\n")

        # Number of lines
        self._n_lines = len(self._lines)
    # end clean_text

# end TextSampling
