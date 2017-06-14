import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from tqdm import tqdm

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

    # visible layer:output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # output
    y_output = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_output))

    return y_output,cross_entropy

def new_model(type,data_dir="/tmp/tensorflow/mnist/input_data"):
    if(type=="tensorflow"):
        return TF_Model(data_dir)

class Model():
    def __init__(self):
        pass

    def save_model();
        pass

    def load_model():
        pass

class TF_Model(Model):
    def __init__(self,data_dir="/tmp/tensorflow/mnist/input_data"):
        self.data_set = input_data.read_data_sets(data_dir, one_hot=True)

        self.model=self.build_model()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def build_model(self):
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.y_output,self.objective_function=tf_LeNet(self.x,self.y_)
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.objective_function)

    def train(self,epoch_num=100):
        # Train
        print("training: epoch number",epoch_num)
        for _ in tqdm(range(epoch_num)):
            batch_xs, batch_ys = self.data_set.train.next_batch(100)
            # self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys,self.keep_prob:0.1})
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        # self.save_model("tensorflow_model.h5py")

    def test(self):
        # Test trained model
        self.correct_prediction = tf.equal(tf.argmax(self.y_output, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        print("accuracy is ",self.sess.run(self.accuracy, feed_dict={self.x: self.data_set.test.images, self.y_: self.data_set.test.labels}))
