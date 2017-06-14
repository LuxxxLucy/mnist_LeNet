import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape,init_value=0.1):
    initial = tf.constant(init_value , shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def tf_LeNet(x,y_):
    # input 28*28=784
    # reshape for conv
    x_image = tf.reshape(x, [-1,28,28,1])

    # first layer:conv
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer:conv
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # third layer:fully-connected
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # add dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # visible layer:output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # output
    y_output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_output))

    return y_output,cross_entropy

class Model:
    def __init__(self,framework_name,mnist=mnist):
        self.model=tensorflow_model()
        self.mnist=mnist

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def tensorflow_model(self):
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.y_output,objective_function=tf_LeNet(x,y_)
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.objective_function)

    def train(self):
        # Train
        for _ in range(1000):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    def test(self):
        # Test trained model
        self.correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels}))
