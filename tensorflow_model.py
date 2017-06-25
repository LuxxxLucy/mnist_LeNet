import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data

import utility

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

def max_pool_1x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

def variable_summaries(var):
    """
        Attach a lot of summaries to a Tensor
        (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
        Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

def tf_LeNet(x,y_):
    # input 28*28=784
    # reshape for conv
    with tf.name_scope('input_reshape_image'):
        x_image = tf.reshape(x, [-1,28,28,1])
        # image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 10)

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

    return y_output

def loss(y_predition,y_label):

    """

    Calculates the loss from the logits and the labels.
    Args:
        y_predition: Logits tensor, float - [batch_size, NUM_CLASSES].
        y_labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.

    """

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_predition)
        with tf.name_scope('total'):
            loss = tf.reduce_mean(diff)
    return loss


class TF_Model():
    def __init__(self,FLAGS):
        self.data_set = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        self.path=FLAGS.path+"_tf_sess.ckpt"
        self.log_dir=FLAGS.log_path
        self.FLAGS=FLAGS
        if(FLAGS.load!=False):
            # load a existing model
            # self.inference()
            self.load_model()
            self.simple_log=utility.load_json(self.log_dir+"simple_log.json")
            self.current_i=self.simple_log["current_i"]
        else:
            # which means build a new model
            print("build a new model")
            self.x,self.y=self.build_graph_input()
            self.current_i=0
            self.simple_log=dict()

            self.inference(self.x,self.y)
            self.current_i=0
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            self.saver = tf.train.Saver()
            utility.check_dir(self.log_dir)
            self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
            self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')

    def load_model(self):
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver= tf.train.import_meta_graph(self.path+".meta")
        print("loading existing model from ",self.path)
        self.saver.restore(self.sess,self.path)

        self.merged = tf.summary.merge_all()
        print("loading success")
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')
        return

    def save_model(self):
        path=self.saver.save(self.sess,self.path)
        try:
            with open(self.log_dir+"simple_log.json","r") as f:
                d=json.load(f)
        except:
            d=dict()

        d["current_i"]=self.current_i

        utility.save_json(d,self.log_dir+"simple_log.json")

        print("model saved in path ",path)
        return


    def build_graph_input(self):
        with tf.name_scope('input'):
            # Create the model
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            # Define loss and optimizer
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        return self.x,self.y_

    def inference(self,x,y_):
        '''

        build the inference graph

        '''

        self.y_output=tf_LeNet(x,y_)

        self.objective_function=loss(self.y_output,self.y_)

        tf.summary.scalar('cross_entropy', self.objective_function)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.objective_function)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.y_output, 1), tf.argmax(self.y_, 1))
            with tf.name_scope('accuracy'):
                # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

        return


    def train(self,epoch_num=100):
        # Train
        print("training: epoch number",epoch_num)
        for i in utility.logged_range(epoch_num,log_info="training!"):
            batch_xs, batch_ys = self.data_set.train.next_batch(100)
            # self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys,self.keep_prob:0.1})

            if i % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = self.sess.run([self.merged, self.accuracy], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                self.test_writer.add_summary(summary, i+self.current_i)

            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    summary, _ = self.sess.run([self.merged, self.train_step], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                    self.train_writer.add_summary(summary, i+self.current_i)
                else:  # Record a summary
                    summary, _ = self.sess.run([self.merged, self.train_step], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                    self.train_writer.add_summary(summary, i+self.current_i)
        self.current_i+=epoch_num

        self.train_writer.close()
        self.test_writer.close()

        self.save_model()
        self.test()

    def test(self):
        # Test trained model
        print("now start evaluating the trained model")
        acc=self.sess.run(self.accuracy, feed_dict={self.x: self.data_set.test.images, self.y_: self.data_set.test.labels})
        print("accuracy is ",acc)
        try:
            self.simple_log=utility.load_json(self.log_dir+"simple_log.json")
        except:
            print("simple log does not exist, create a new one")
            self.simple_log=dict()


        self.simple_log["current_i"]=self.current_i
        self.simple_log["acc at "+str(self.current_i)]=float(acc)
        utility.save_json(self.simple_log,self.log_dir+"simple_log.json")
