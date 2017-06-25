#!/usr/bin/env python
#
import os,random
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *

from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm

def build_graph_model():
    # Build graph model ...
    nch = 200
    g_input = Input(shape=[100])
    H = Dense(nch*14*14, init='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Reshape( [ 14, 14,nch] )(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    H = Reshape( [ 1,28,28] )(H)
    g_V = Activation('sigmoid')(H)
    model = Model(g_input,g_V)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):
        # real image
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        # generative image
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)


if __name__ == "__main__":
    nb_classes = 10

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model=build_graph_model()

    model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=4,
          show_accuracy=True, verbose=1,
          validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
