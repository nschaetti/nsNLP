# -*- coding: utf-8 -*-
#
# File : corpus/IQLACorpus.py
# Description : .
# Date : 16/08/2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
from Sample import Sample
import codecs


# Class to access to a text
class Text(Sample):
    """
    Class to access to a text
    """

    # Constructor
    def __init__(self, text_path, author, text_title):
        """
        Constructor
        :param text_path:
        :param author:
        """
        super(Text, self).__init__(text_path, author)
        self._text_path = text_path
        self._author = author
        self._title = text_title
    # end __init__

    ########################################
    # Public
    ########################################

    # Get title
    def get_title(self):
        """
        Get title
        :return:
        """
        return self._title
    # end get_title

    # Get text
    def get_text(self):
        """
        Get text
        :return:
        """
        return codecs.open(self._text_path, 'r', encoding='utf-8').read()
    # end text

    # Get author
    def get_author(self):
        """
        Get author
        :return:
        """
        return self._author
    # end author

    # Get path
    def get_path(self):
        """
        Get path
        :return:
        """
        return self._text_path
    # end get_path

    # Get X
    def x(self):
        """
        Get X
        :return:
        """
        return self.get_text()
    # end x

    # Get Y
    def y(self):
        """
        Get Y
        :return:
        """
        return self.get_author().get_name()
    # end y

    ########################################
    # Override
    ########################################

    # To string
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"Text(title: {}, path:{}, author:{})".format(self._title, self._text_path, self._author.get_name())
    # end __unicode__

    # To string
    def __str__(self):
        """
        To string
        :return:
        """
        return "Text(title: {}, path:{}, author:{})".format(self._title, self._text_path, self._author.get_name())
    # end __unicode__

# end Text
