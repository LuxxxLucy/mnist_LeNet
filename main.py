from __future__ import absolute_import
from __future__ import division
from pprint import pprint as pr

import argparse
import sys

import tensorflow as tf
import keras
# import caffe
# import pytorch

import tensorflow_model

FLAGS = None

def new_model(FLAGS):
    if(FLAGS.framework=="tensorflow"):
        return tensorflow_model.TF_Model(FLAGS)
    elif FLAGS.framework=="pytorch":
        pass
    else:
        print("illegal option for framework!!!")
        print("exit with error")
        quit()

def main_access(_):
    model=new_model(FLAGS)
    if FLAGS.train:
        model.train(FLAGS.train_num)
    if FLAGS.test:
        model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    parser.add_argument('-l', '--load', action='store_true', help='load a existing model or not')
    parser.add_argument('-t', '--train', action='store_true', help='train or not')
    parser.add_argument('-e', '--test', action='store_true', help='test a existing model or not')
    parser.add_argument('--train_num', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--path', type=str, default='model/model', help='specify the model you want to load')
    parser.add_argument('--log_path', type=str, default='log/', help='specify the path of log file(tensorboard)')
    parser.add_argument('--framework', type=str, default='tensorflow', help='frame work name \'tensorflow\' \'keras\'  \'caffe\' or \'pytorch\' ')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main_access, argv=[sys.argv[0]] + unparsed)
