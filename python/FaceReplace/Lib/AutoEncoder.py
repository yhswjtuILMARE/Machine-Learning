'''
Created By ILMARE
@Date 2019-2-27
'''

import tensorflow as tf
import numpy as np
import sklearn
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import cv2
from tensorflow.python.client import device_lib

def xavier_init(input_count, output_count, constant=1):
    low = -constant * np.sqrt(6.0 / (input_count + output_count))
    high = constant * np.sqrt(6.0 / (input_count + output_count))
    return tf.random_uniform((input_count, output_count),
                            minval=low, maxval=high,
                            dtype=tf.float32)

def standard_scale(X_train, X_test):
    pre = sklearn.preprocessing.StandardScaler().fit(X_train)
    return pre.transform(X_train), pre.transform(X_test)

def get_random_block_from_data(data, batch_size):
    start_idx = np.random.randint(0, len(data) - batch_size)
    return data[start_idx: start_idx + batch_size, :]

def conv2d(X, W, strides=[1,1,1,1], padding="VALID"):
    return tf.nn.conv2d(X, W, strides=strides, padding=padding)

class AutoEncoder:
    def __init__(self, learning_rate, max_step, batch_size, channel=3):
        self._channel = channel
        self._learning_rate = learning_rate
        self._max_step = max_step
        self._batch_size = batch_size
        self.define_network()
    def define_network(self):
        self._x = tf.placeholder(shape=[None, 64, 64, self._channel], dtype=tf.float32)
        def define_encoder():
            with tf.name_scope("conv1"):
                conv1_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [5,5,self._channel,128], stddev=0.1, dtype=tf.float32))
                h_conv1 = tf.nn.conv2d(self._x, conv1_kernal, strides=[1,2,2,1], padding="SAME")
                conv1_b = tf.Variable(initial_value=tf.zeros([128], dtype=tf.float32))
                conv1 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1, conv1_b), alpha=0.1)
            with tf.name_scope("conv2"):
                conv2_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [5,5,128,256], stddev=0.1, dtype=tf.float32))
                h_conv2 = tf.nn.conv2d(conv1, conv2_kernal, strides=[1,2,2,1], padding="SAME")
                conv2_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
                conv2 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2, conv2_b), alpha=0.1)
            with tf.name_scope("conv3"):
                conv3_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [5,5,256,512], stddev=0.1, dtype=tf.float32))
                h_conv3 = tf.nn.conv2d(conv2, conv3_kernal, strides=[1,2,2,1], padding="SAME")
                conv3_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
                conv3 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv3, conv3_b), alpha=0.1)
            with tf.name_scope("conv4"):
                conv4_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [5,5,512,1024], stddev=0.1, dtype=tf.float32))
                h_conv4 = tf.nn.conv2d(conv3, conv4_kernal, strides=[1,2,2,1], padding="SAME")
                conv4_b = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                conv4 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv4, conv4_b), alpha=0.1)
                flatten1 = tf.layers.Flatten()(conv4)
                tmp_dim = int(flatten1.get_shape()[1])
            with tf.name_scope("dense1"):
                w1 = tf.Variable(initial_value=xavier_init(tmp_dim, 1024))
                b1 = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                link1 = tf.nn.bias_add(tf.matmul(flatten1, w1), b1)
            with tf.name_scope("dense2"):
                w2 = tf.Variable(initial_value=xavier_init(1024, tmp_dim))
                b2 = tf.Variable(initial_value=tf.zeros([tmp_dim], dtype=tf.float32))
                link2 = tf.nn.bias_add(tf.matmul(link1, w2), b2)
                link2 = tf.reshape(link2, [-1, 4, 4, 1024])
            with tf.name_scope("upscale1"):
                upscale1_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [3, 3, 1024, 2048], stddev=0.1, dtype=tf.float32))
                upscale1_b = tf.Variable(initial_value=tf.zeros([2048], dtype=tf.float32))
                h_upscale1 = tf.nn.conv2d(link2, upscale1_kernal, strides=[1, 1, 1, 1], padding="SAME")
                upscale1 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale1, upscale1_b), alpha=0.1)
                upscale1 = tf.depth_to_space(upscale1, 2)
                self._hidden = upscale1
        def define_decoder():
            with tf.name_scope("upscale2"):
                upscale2_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [3,3,512,1024], stddev=0.1, dtype=tf.float32))
                upscale2_b = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                h_upscale2 = tf.nn.conv2d(self._hidden, upscale2_kernal, strides=[1,1,1,1], padding="SAME")
                upscale2 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale2, upscale2_b), alpha=0.1)
                upscale2 = tf.depth_to_space(upscale2, 2)
            with tf.name_scope("upscale3"):
                upscale3_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [3,3,256,512], stddev=0.1, dtype=tf.float32))
                upscale3_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
                h_upscale3 = tf.nn.conv2d(upscale2, upscale3_kernal, strides=[1,1,1,1], padding="SAME")
                upscale3 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale3, upscale3_b), alpha=0.1)
                upscale3 = tf.depth_to_space(upscale3, 2)
            with tf.name_scope("upscale4"):
                upscale4_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [3,3,128,256], stddev=0.1, dtype=tf.float32))
                upscale4_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
                h_upscale4 = tf.nn.conv2d(upscale3, upscale4_kernal, strides=[1,1,1,1], padding="SAME")
                upscale4 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale4, upscale4_b), alpha=0.1)
                upscale4 = tf.depth_to_space(upscale4, 2)
            with tf.name_scope("conv5"):
                conv5_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [5, 5, 64, self._channel], stddev=0.1, dtype=tf.float32))
                h_conv5 = tf.nn.conv2d(upscale4, conv5_kernal, strides=[1, 1, 1, 1], padding="SAME")
                conv5_b = tf.Variable(initial_value=tf.zeros([self._channel], dtype=tf.float32))
                conv5 = tf.nn.sigmoid(tf.nn.bias_add(h_conv5, conv5_b))
                self._reconstruct = conv5
        define_encoder()
        define_decoder()
    def define_loss(self):
        self.loss = tf.reduce_mean(tf.pow(tf.subtract(self._x, self._reconstruct), 2))
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)


class ConvolutionalAutoencoder:
    def __init__(self, learning_rate, max_step, batch_size):
        self._learning_rate = learning_rate
        self._max_step = max_step
        self._batch_size = batch_size
        self._sess = tf.Session()
        self.define_network()
        self.define_loss()
    def define_network(self):
        self._x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        def define_encoder():
            with tf.variable_scope("conv1") as scope:
                conv1_kernal = tf.get_variable(name="conv1_kernal", initializer=tf.truncated_normal(
                    shape=[5, 5, 1, 16], stddev=0.1, dtype=tf.float32))
                conv1_biases = tf.get_variable(name="conv1_biases", initializer=tf.zeros(shape=[16], dtype=tf.float32))
                h_conv1 = tf.nn.conv2d(self._x, conv1_kernal, strides=[1, 2, 2, 1], padding="SAME")
                conv1 = tf.nn.relu(tf.add(h_conv1, conv1_biases))
            with tf.variable_scope("conv2") as scope:
                conv2_kernal = tf.get_variable(name="conv2_kernal", initializer=tf.truncated_normal(
                    shape=[5, 5, 16, 32], stddev=0.1, dtype=tf.float32))
                conv2_biases = tf.get_variable(name="conv2_biases", initializer=tf.zeros(shape=[32], dtype=tf.float32))
                h_conv2 = tf.nn.conv2d(conv1, conv2_kernal, strides=[1, 2, 2, 1], padding="SAME")
                conv2 = tf.nn.relu(h_conv2 + conv2_biases)
                self._hidden = conv2
        def define_decoder():
            with tf.variable_scope("uconv1") as scope:
                uconv1_kernal = tf.get_variable(name="uconv1_kernal", initializer=tf.truncated_normal(
                    shape=[5, 5, 32, 64], dtype=tf.float32, stddev=0.1))
                uconv1_biases = tf.get_variable(name="uconv1_biases",
                                                initializer=tf.zeros(shape=[64], dtype=tf.float32))
                h_uconv1 = tf.nn.conv2d(self._hidden, uconv1_kernal, strides=[1, 1, 1, 1], padding="SAME")
                uconv1 = tf.nn.relu(tf.depth_to_space(tf.nn.bias_add(h_uconv1, uconv1_biases), 2))
            with tf.variable_scope("uconv2") as scope:
                uconv2_kernal = tf.get_variable(name="uconv2_kernal", initializer=tf.truncated_normal(
                    shape=[5, 5, 16, 4], stddev=0.1, dtype=tf.float32))
                uconv2_biases = tf.get_variable(name="uconv2_biases",
                                                initializer=tf.zeros(shape=[4], dtype=tf.float32))
                h_uconv2 = tf.nn.conv2d(uconv1, uconv2_kernal, strides=[1, 1, 1, 1], padding="SAME")
                uconv2 = tf.depth_to_space(tf.nn.bias_add(h_uconv2, uconv2_biases), 2)
                self._reconstruct = uconv2
        define_encoder()
        define_decoder()
    def define_loss(self):
        self._loss = tf.reduce_sum(tf.pow(tf.subtract(self._x, self._reconstruct), 2))
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)
    def get_hidden_output(self, X):
        return self._sess.run(self._hidden, feed_dict={self._x: X})
    def get_reconstruct(self, X):
        return self._sess.run(self._reconstruct, feed_dict={self._x: X})
    def generate(self, hidden=None):
        if hidden is None:
            raise Exception("There is no hidden input")
        return self._sess.run(self._reconstruct, feed_dict={self._hidden: hidden})
    def train(self):
        self._sess.run(tf.global_variables_initializer())
        mnist = input_data.read_data_sets(r"F:\tensorflow\MNIST_DATA", one_hot=True)
        n_examples = int(mnist.train.num_examples)
        for step in range(self._max_step):
            avg_cost = 0
            max_batch = n_examples // self._batch_size
            for idx in range(max_batch):
                train = get_random_block_from_data(mnist.train.images, self._batch_size)
                train = np.reshape(train, newshape=[-1, 28, 28, 1])
                _, loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._x: train})
                avg_cost += (loss / max_batch)
                if (idx % 50) == 0:
                    print("cost: %.3f" % loss, " batch: ", idx)
            print("avg_cost: %.3f" % avg_cost, " step: ", step)
        tf.train.Saver().save(self._sess, save_path="F:/tensorflow/automodel/model1/encoder")
    def load_model(self):
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint("F:/tensorflow/automodel/model1/"))
        mnist = input_data.read_data_sets(r"F:\tensorflow\MNIST_DATA", one_hot=True)
        source = np.reshape(mnist.test.images[15], [1, 28, 28, 1])
        dest = self.get_reconstruct(source)
        source = np.reshape(source, [28, 28])
        dest = np.reshape(dest, [28, 28])
        print(source.shape, dest.shape)
        fig = plt.figure("test")
        ax = fig.add_subplot(121)
        ax.imshow(source)
        bx = fig.add_subplot(122)
        bx.imshow(dest)
        plt.show()



class AGNAutoEncoder:
    def __init__(self, input_size, hidden_size,
                 transfer_function=tf.nn.softplus,learning_rate=0.001,
                 scale=0.1, batch_size=128, max_step=20):
        self._learning_tate = learning_rate
        self._max_step = max_step
        self._batch_size = batch_size
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._transfer = transfer_function
        self._scale = tf.placeholder(tf.float32)
        self._training_scale = scale
        self._weight = self.init_weights()
        self._sess = tf.Session()
        self.defineNetwork()
    def defineNetwork(self):
        self._x = tf.placeholder(shape=[None, self._input_size], dtype=tf.float32)
        self._hidden = self._transfer(tf.add(tf.matmul(
            self._x + self._training_scale * tf.random_normal((self._input_size, )),
            self._weight['w1']), self._weight['b1']))
        self._reconstruction = tf.add(tf.matmul(self._hidden,
                                                self._weight['w2']), self._weight['b2'])
        self._loss = 0.5 * tf.reduce_sum(tf.pow(
            tf.subtract(self._reconstruction, self._x), 2.0))
        self._optimizer = tf.train.AdamOptimizer(self._learning_tate).minimize(self._loss)
    def init_weights(self):
        all_weight = dict()
        all_weight['w1'] = tf.Variable(initial_value=xavier_init(self._input_size,
                                                                 self._hidden_size), dtype=tf.float32)
        all_weight['b1'] = tf.Variable(initial_value=tf.zeros(shape=[self._hidden_size],
                                                              dtype=tf.float32))
        all_weight['w2'] = tf.Variable(initial_value=xavier_init(self._hidden_size,
                                                                 self._input_size), dtype=tf.float32)
        all_weight['b2'] = tf.Variable(initial_value=tf.zeros(shape=[self._input_size],
                                                              dtype=tf.float32))
        return all_weight
    def part_fit(self, X):
        _loss, _ = self._sess.run([self._loss, self._optimizer],
                                  feed_dict={self._x: X, self._scale: self._training_scale})
        return _loss
    def calculate_total_cost(self, X):
        _loss = self._sess.run([self._loss],
                               feed_dict={self._x: X, self._scale: self._training_scale})
        return _loss
    def transform(self, X):
        return self._sess.run(self._hidden, feed_dict={self._x: X,
                                                       self._scale: self._training_scale})
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self._weight['b1'])
        return self._sess.run(self._reconstruction, feed_dict={self._hidden: hidden})
    def reconstrct(self, X):
        return self._sess.run(self._reconstruction, feed_dict={self._x: X,
                                                               self._scale: self._training_scale})
    def getWeight(self):
        return self._sess.run(self._weight['w1'])
    def getBiases(self):
        return self._sess.run(self._weight['b1'])
    def train(self):
        self._sess.run(tf.global_variables_initializer())
        mnist = input_data.read_data_sets("/home/ilmare/dataSet/mnist", one_hot=True)
        train, _ = standard_scale(mnist.train.images, mnist.test.images)
        n_examples = int(mnist.train.num_examples)
        for idx in range(self._max_step):
            avg_cost = 0
            total_batch = n_examples // self._batch_size
            for batch in range(0, total_batch):
                train_tmp = get_random_block_from_data(train, self._batch_size)
                _cost = self.part_fit(train_tmp)
                avg_cost += _cost / n_examples * self._batch_size
            if (idx + 1) % 5 == 0:
                print("avg_cost: %.3f" % avg_cost, " step: ", idx)
        tf.train.Saver().save(self._sess, save_path="{0}autoencoder".format("/home/ilmare/Desktop/FaceReplace/model/"))
    def load_model(self):
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint("/home/ilmare/Desktop/FaceReplace/model/"))
        mnist = input_data.read_data_sets("/home/ilmare/dataSet/mnist", one_hot=True)
        source = np.reshape(mnist.train.images[0], [1, 784])
        dest = self.reconstrct(source)
        source = np.reshape(source, [28, 28])
        dest = np.reshape(dest, [28, 28])
        print(source.shape, dest.shape)
        # fig = plt.figure("test")
        # ax = fig.add_subplot(121)
        # ax.imshow(source)
        # bx = fig.add_subplot(122)
        # bx.imshow(dest)
        # plt.show()
        cv2.imshow("lalal", dest)
        cv2.waitKey(0)


if __name__ == "__main__":
    obj = ConvolutionalAutoencoder(0.01, 20, 64)
    obj.load_model()
    # obj = AGNAutoEncoder(784, 200, scale=0.01,max_step=100,
    #                      learning_rate=0.001)
    # obj.load_model()