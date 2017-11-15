# -*- coding: utf-8 -*-
#
# File : core/downloader/PySpeechesConfig.py
# Description : .
# Date : 20th of February 2017
#
# This file is part of nsNLP.  nsNLP is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import torch.nn as nn
import torch.nn.functional as F


# Convolution module
class ConvNet(nn.Module):
    """
    Convolution module
    """

    # Constructor
    def __init__(self, n_classes=2, channels=(1, 10, 2, 20, 2, 4800, 50), kernel_size=5, stride=1):
        """

        :param n_classes:
        :param params:
        """
        super(ConvNet, self).__init__()

        # Max pooling size
        self._max_pool1_size = channels[2]
        self._max_pool2_size = channels[4]

        # 2D convolution layer, 1 in channel, 10 out channels (filters), kernel size 5
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=stride)

        # 2D convolution layer, 10 input channels, 20 output channels (filters), kernel size 5
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size)

        # 2D Dropout layer, with probability of an element to be zeroed to 0.5
        self.conv2_drop = nn.Dropout2d()

        # Linear transformation with 4800 inputs features and 50 output features
        self.fc1 = nn.Linear(channels[3], channels[4])

        # Linear transformation with 50 inputs features and 2 output features
        self.fc2 = nn.Linear(channels[4], n_classes)
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # ReLU << Max pooling 2D with kernel size 2 << Convolution layer 1 << x
        x = self.conv1(x)
        x = F.max_pool2d(x, self._max_pool1_size)
        x = F.relu(x)

        # ReLU << Max pooling 2D with kernel size 2 << Dropout 2D << Convolution layer 2 << x
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), self._max_pool2_size))

        # Put all 320 features into 1D line << x
        x = x.view(-1, ConvNet.num_flat_features(x))

        # ReLU << Linear model on 4800 features to 50 outputs << x
        x = F.relu(self.fc1(x))

        # Trained dropout << x
        x = F.dropout(x, training=self.training)

        # Linear model on 50 features to 2 outputs
        x = self.fc2(x)

        # Softmax layer << x
        return F.log_softmax(x)
    # end forward

    ##############################################
    # Static
    ##############################################

    # Number of flat features
    @staticmethod
    def num_flat_features(x):
        """
        Number of flat features
        :param x:
        :return:
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    # end num_flat_features

# end ConvNet
