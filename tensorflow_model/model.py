"""
The core model
"""

import settings
from tensorflow_model.nn import *
from tensorflow_model.model_session import ModelSession

import keras
from keras.layers import Dense,Dropout

savePath = settings.MODEL_STORE_PATH

ITEM_DIM = 100


class Lenet_Model_Session(ModelSession):
    """
    An OOP style ModelSession for fully_connected model (our first model)
    """

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new model session.

        :param kwargs: optional graph parameters
        :type kwargs: dict
        :return: new model session
        :rtype: ModelSession
        """
        session = tf.Session()
        from keras import backend as K
        K.set_session(session)
        with session.graph.as_default():
            cls.create_graph(**kwargs)
        session.run(tf.global_variables_initializer())
        return cls(session, tf.train.Saver(), kwargs['args'])

    @staticmethod
    def create_graph(class_num, item_dim, args, val_portion=0.4, save_path=savePath, n_epoch=5,
                 batch_size=128, learning_rate=0.00001, print_freq=100, global_step=0,
                 layer_num=1, top_unit_num=32, dropout=0.0):
        """
        The fully connected model for predicting next item that would be consumed by the user

        :param class_num: The class number, which means the number of different films' types
        :param item_dim: The item's dimension after embedding
        :param val_portion: Determine the number of validation data
        :param save_path: The path to save the model
        :param n_epoch: The number of epochs for training
        :param batch_size: The number of records in each batch
        :param learning_rate: The learning rate
        :param print_freq: The frequency to print the information
        :param global_step: The global step used for continuous training
        :param layer_num: The number of fully-connected NN layer
        :param top_unit_num: The number of units in the top layer of this NN
        :param dropout: The dropout rate of each layer

        """

        IMAGE_DIMENSION=784


        iteration = tf.Variable(initial_value=0, trainable=False, name="iteration")
        with tf.variable_scope("parameters"):
            x = tf.placeholder(tf.float32, shape=[None, IMAGE_DIMENSION], name='x')
            y = tf.placeholder(tf.int64, shape=[None,10], name='y')
            drop_rate = tf.placeholder(tf.float32, name="drop_rate")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        input_layer = tf.reshape(x, [-1, 28, 28, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
          inputs=dense, rate=drop_rate )

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)
        y_ = tf.nn.softmax(logits, name='y_result')

        with tf.variable_scope("train"):
            # onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)
            cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_)

            train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                              use_locking=False).minimize(cost, global_step=iteration, name="train_step")
            y_predict = tf.argmax(y_, 1)
            y=tf.argmax(y,1)
            correct_prediction = tf.equal(y_predict, y)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    def __str__(self):
        return "tensorflow LeNet Model(iteration %d)" % (
            self.session.run(self.iteration))

    def preprocess(self, X, y):
        '''
        potential preoporcess. Not necessary in this model
        '''
        return X, y

    def train(self, X_train_a, y_train_a, learning_rate, drop_rate=0.0):
        x, y = self.preprocess(X_train_a, y_train_a)
        return self.session.run([self.train_step, self.iteration],
                                feed_dict={self.x: x,
                                           self.y: y,
                                           self.drop_rate: drop_rate,
                                           self.learning_rate: learning_rate})[1]

    def test(self, x, y):
        x, y = self.preprocess(x, y)
        result= self.session.run(self.accuracy, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0})
        return result

    def test_batch(self, x, y):
        x, y = self.preprocess(x, y)
        result= self.session.run(self.accuracy, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0})
        return result

    @property
    def train_step(self):
        return self._tensor("train/train_step:0")

    @property
    def accuracy(self):
        return self._tensor("train/accuracy:0")

    @property
    def y_(self):
        return self._tensor("y_result:0")

    @property
    def iteration(self):
        return self._tensor("iteration:0")

    @property
    def x(self):
        return self._tensor("parameters/x:0")

    @property
    def y(self):
        return self._tensor("parameters/y:0")

    @property
    def drop_rate(self):
        return self._tensor("parameters/drop_rate:0")

    @property
    def learning_rate(self):
        return self._tensor("parameters/learning_rate:0")

    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)
