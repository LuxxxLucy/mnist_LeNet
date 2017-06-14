from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


import tensorflow as tf

import models

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    model=models.new_model(FLAGS.framework)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--framework', type=str, default='tensorflow',
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
