"""
data loader for MNIST

"""


import numpy as np
import pickle
from os import path
from pathlib import Path
import random

import settings

MNIST_PATH = path.join(settings.DATA_STORE_PATH, 'mnist')

class DataLoader(object):
    """ an object that generates batches of MovieLens_1M data for training """

    def __init__(self, args):
        """
        Initialize the DataLoader
        :param args: all kinds of argument
        """

        self.data_dir = MNIST_PATH

        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(self.data_dir, one_hot=True)

        self.X_train = mnist.train.images
        self.y_train = mnist.train.labels

        self.X_val = mnist.test.images
        self.y_val = mnist.test.labels

        print(self.X_train[0].shape)
        print(self.y_train[0].shape)
        print(self.X_val[0].shape)
        print(self.y_val[0].shape)
