from __future__ import absolute_import
from __future__ import division
from pprint import pprint as pr

import argparse
import sys

import tensorflow as tf
import keras
# import caffe
# import pytorch

import models

FLAGS = None


def main(_):
    model=models.new_model(FLAGS.framework)
    model.train()
    model.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    parser.add_argument('--framework', type=str, default='tensorflow', help='frame work name \'tensorflow\' \'keras\'  \'caffe\' or \'pytorch\' ')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
