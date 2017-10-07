from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math
import time


batch = tf.Variable(0)
batch_size = 256

training_epoch = 10
training_rate_ = 9.995e-4
training_decay_ = 0.895

start_time = time.time()  # clocking start

mnist = read_data_sets("./MNIST_data/", one_hot=True)


class CnnModel:
    def __init__(self, s, name):
        tf.set_random_seed(777)  # reproducibility

        self.s = s
        self.name = name
        self.build_cnn()

    def conv2d(self, x, filter_, ksize_=5, stride_=1, activation_=tf.nn.relu):
        return tf.layers.conv2d(inputs=x, filters=filter_, kernel_size=[ksize_, ksize_], strides=stride_, padding="SAME", activation=activation_)

    def max_pool(self, conv_, ksize_=2, stride_=2):
        return tf.layers.max_pooling2d(inputs=conv_, pool_size=[ksize_, ksize_], padding="SAME", strides=stride_)

    def build_cnn(self):
        with tf.name_scope(self.name):
            # data placeholder
            self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X-input')  # 28 * 28 * 1 image
            self.Y = tf.placeholder(tf.int32, shape=[None, 10], name='Y-input')  # 0 ~ 9

            # dropout_rate placeholder
            self.p_keep_conv = tf.placeholder("float")
            self.p_keep_hidden = tf.placeholder("float")

            # learning_rate & decay placeholder
            self.training_rate = tf.placeholder(tf.float32)
            self.training_decay = tf.placeholder(tf.float32)

            with tf.name_scope("cnn_layer_1"):
                conv1 = self.conv2d(self.X, 32)
                pool1 = self.max_pool(conv1)
                layer1 = tf.layers.dropout(pool1, self.p_keep_conv)

            with tf.name_scope("cnn_layer_2"):
                conv2 = self.conv2d(layer1, 64)
                layer2 = tf.layers.dropout(conv2, self.p_keep_conv + 0.05)

            with tf.name_scope("cnn_layer_3"):
                conv3 = self.conv2d(layer2, 128, stride_=2, activation_=tf.nn.elu)
                layer3 = tf.reshape(conv3, [-1, 7 * 7 * 128])
                layer3 = tf.layers.dropout(layer3, self.p_keep_conv + 0.1)

            with tf.name_scope("fully_connected_1"):
                fc1 = tf.layers.dense(layer3, units=1024, activation=tf.nn.elu)
                layer4 = tf.layers.dropout(fc1, self.p_keep_hidden)

            with tf.name_scope("hypothesis"):
                self.logits = tf.layers.dense(layer4, units=10)

        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits))

        with tf.name_scope("train"):
            learning_rate = tf.train.exponential_decay(
                self.training_rate,
                batch * batch_size,
                mnist.train.num_examples,
                self.training_decay,
                staircase=True
            )
            self.train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=batch)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict_(self, x):
        return self.s.run(self.logits, feed_dict={self.X: x, self.p_keep_conv:0., self.p_keep_hidden:0., self.training_decay:1.})

    def acc_(self, x, y):
        return self.s.run(self.accuracy, feed_dict={self.X: x, self.Y: y, self.p_keep_conv:0., self.p_keep_hidden:0., self.training_decay:1.})

    def train_(self, x, y, train_rate_, train_decay_):
        return self.s.run([self.cost, self.train], feed_dict={self.X: x, self.Y: y, self.p_keep_conv:0.2, self.p_keep_hidden:0.5, self.training_rate:train_rate_, self.training_decay:train_decay_})


class RnnModel:
    def __init__(self, s, name):
        tf.set_random_seed(778)  # reproducibility

        self.s = s
        self.name = name
        self.build_rnn()

    def xavier_init(self, n_inputs, n_outputs, uniform=True):
        if uniform:
            init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
            return tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
            return tf.truncated_normal_initializer(stddev=stddev)

    def build_rnn(self):
        with tf.name_scope(self.name):
            # data placeholder
            self.X = tf.placeholder(tf.float32, shape=[None, 28, 28], name='X-input')  # 28 * 28 * 1 image
            self.Y = tf.placeholder(tf.int32, shape=[None, 10], name='Y-input')  # 0 ~ 9

            # Weight & Bias
            self.Win = tf.get_variable('Win', shape=[28, 128], initializer=self.xavier_init(28, 128))
            self.Wout = tf.get_variable('Wout', shape=[128, 10], initializer=self.xavier_init(128, 10))

            self.bin = tf.Variable(tf.constant(0., shape=[128]))
            self.bout = tf.Variable(tf.constant(0., shape=[10]))

            # x => [128 * 28, 28]
            self.X_in = tf.nn.bias_add(tf.matmul(tf.reshape(self.X, [-1, 28]), self.Win), self.bin)
            self.X_in = tf.reshape(self.X_in, [-1, 28, 128])  # X_in => [128, 28, 128]

            with tf.name_scope("LSTM_cell"):
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=128)

                init_state = cell.zero_state(batch_size, dtype=tf.float32)

                outputs, _ = tf.nn.dynamic_rnn(cell, self.X_in, initial_state=init_state, dtype=tf.float32, time_major=False)
                outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

                self.pred = tf.nn.bias_add(tf.matmul(outputs[-1], self.Wout), self.bout)

            with tf.name_scope("cost"):
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.Y))

            with tf.name_scope("train"):
                learning_rate = tf.train.exponential_decay(
                    training_rate_,
                    batch * batch_size,
                    mnist.train.num_examples,
                    training_decay_,
                    staircase=True
                )
                self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

            with tf.name_scope("prediction"):
                self.predict = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
                self.acc = tf.reduce_mean(tf.cast(self.predict, tf.float32))

    def train_(self, x, y):
        return self.s.run([self.cost, self.train], feed_dict={self.X: x, self.Y: y})

    def acc_(self, x, y):
        return self.s.run(self.acc, feed_dict={self.X: x, self.Y: y})


# MODEL & CHECKPOINT directory
model_dir = "./mnist/mnist.ckpt"
train_dir = "./mnist/train/"
check_point_dir = "./mnist/checkpoint/"

# training config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

s = tf.Session(config=config)

models = []
cnn_model_size = 1  # 3, 5 are more better than 2  # but because of gpu...
rnn_model_size = 1  # 3

for i in range(cnn_model_size):
    models.append(CnnModel(s, "model" + str(i)))

for i in range(rnn_model_size):
    models.append(RnnModel(s, "model" + str(cnn_model_size + i)))


s.run(tf.global_variables_initializer())

# saver & restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(check_point_dir)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(s, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

max_acc = 0.
total_batch = int(mnist.train.num_examples / batch_size)

# model training
for epoch in range(1, training_epoch + 1):
    avg_cost_list, valid_acc_list = np.zeros(len(models)), np.zeros(len(models))

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        for idx, m in enumerate(models):
            if idx < cnn_model_size:
                batch_x = batch_x.reshape(-1, 28, 28, 1)
                test_x = mnist.test.images.reshape(-1, 28, 28, 1)
                c, _ = m.train_(batch_x, batch_y, training_rate_ * (1 + 0.01 * idx), training_decay_ * (1 - 0.005 * idx))
            else:
                batch_x = batch_x.reshape(-1, 28, 28)
                test_x = mnist.test.images.reshape(-1, 28, 28)
                c, _ = m.train_(batch_x, batch_y)

            import random
            r = random.random() % total_batch
            valid_acc_list[idx] = m.acc_(test_x[r * batch_size:(r + 1) * batch_size], mnist.test.labels[r * batch_size:(r + 1) * batch_size])
            avg_cost_list[idx] += c / total_batch

            if valid_acc_list[idx] > max_acc:
                max_acc = valid_acc_list[idx]

                # saving model
                saver.save(s, model_dir)
                print("Max Accuracy : {:.4f}".format(max_acc))

    print("[*] Epoch %03d" % epoch, "Avg Code : ", avg_cost_list, "Max Accuracy : {:.4f}".format(max_acc))
    print("------------------------------------------------------------------")


# Test model and check accuracy
test_size = len(mnist.test.labels)
pred = np.zeros(test_size * 10).reshape(test_size, 10)

for idx, m in enumerate(models):
    if idx > 1:
        test_x = mnist.test.images.reshape(-1, 28, 28)
    else:
        test_x = mnist.test.images.reshape(-1, 28, 28, 1)

    print(idx, 'Accuracy : {:.4f}'.format(m.acc_(test_x, mnist.test.labels)))
    pred += m.predict_(test_x)

ensemble_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_pred, tf.float32))
print('Ensemble Accuracy : ', s.run(ensemble_accuracy))

end_time = time.time() - start_time

# elapsed time
print("[+] Elapsed time {:.10f}s".format(end_time))

s.close()
