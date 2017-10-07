from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import cnn_model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1e-2))


def init_bias(shape):
    return tf.Variable(tf.constant(0., shape=shape))


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


start_time = time.time()  # clocking start

# mnist data
mnist = read_data_sets("./MNIST_data/", one_hot=True)

# MODEL & TRAIN directory
model_ = "./mnist/mnist.ckpt"
train_ = './mnist/train'

# setting random seed
tf.set_random_seed(777)

# data placeholder
X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X-input')  # 28 * 28 * 1 image
Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y-input')  # 0 ~ 9

# dropout_rate placeholder
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# Weights & Biases
W0 = init_weights([5, 5, 1, 32])
W1 = init_weights([5, 5, 32, 64])
W2 = init_weights([5, 5, 64, 128])
W3 = tf.get_variable('W3', shape=[7 * 7 * 128, 1024], initializer=xavier_init(7 * 7 * 128, 1024))
W_last = tf.get_variable('W_last', shape=[1024, 10], initializer=xavier_init(1024, 10))

W = [W0, W1, W2, W3, W_last]

b0 = init_bias([32])
b1 = init_bias([64])
b2 = init_bias([128])
b3 = init_bias([1024])
b_last = init_bias([10])

b = [b0, b1, b2, b3, b_last]

# histogram summary
w0_hist = tf.summary.histogram("weights0", W0)
w1_hist = tf.summary.histogram("weights1", W1)
w2_hist = tf.summary.histogram("weights2", W2)
w3_hist = tf.summary.histogram("weights3", W3)
w_last_hist = tf.summary.histogram("weights_last", W_last)

b0_hist = tf.summary.histogram("bias0", b0)
b1_hist = tf.summary.histogram("bias1", b1)
b2_hist = tf.summary.histogram("bias2", b2)
b3_hist = tf.summary.histogram("bias3", b3)
b_last_hist = tf.summary.histogram("bias_last", b_last)

y_hist = tf.summary.histogram("y", Y)

# model
_X = cnn_model.cnn_model_0(X, W, b, p_keep_conv, p_keep_hidden)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=_X))

batch_size = 50
batch = tf.Variable(0)
with tf.name_scope("train"):
    learning_rate = tf.train.exponential_decay(
        9.95e-4,
        batch * batch_size,
        mnist.train.num_examples,
        0.9,
        staircase=True
    )
    # train = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=batch)
    train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, global_step=batch)

with tf.name_scope("acc"):
    correct_prediction = tf.equal(tf.argmax(_X, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc_hist = tf.summary.histogram("acc", accuracy)

display_step = 200
logging_step = 5

saver = tf.train.Saver()

# training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session() as s:
    # variables init
    s.run(tf.global_variables_initializer())

    # saving the best acc
    max_acc = 0.

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(train_, s.graph)

    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(1, 101):
        avg_cost = 0.

        for i in range(0, total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)  # 28x28x1 input img

            _, train_acc = s.run([train, accuracy], feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv: 0.9, p_keep_hidden: 0.5})
            avg_cost += s.run(cost, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv: 0.9, p_keep_hidden: 0.5}) / total_batch

            # printing result
            if i % display_step == 0:
                print("[*] Epoch %03d" % epoch, "Training Acc : {:.4f}".format(train_acc))

                summary = s.run(merged, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv: 1.0, p_keep_hidden: 1.0})
                writer.add_summary(summary, i)

            if i % logging_step == 0:
                valid_acc = s.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, p_keep_conv: 1.0, p_keep_hidden: 1.0})

                if i % 200 == 0:
                    print("[*] Epoch %03d" % epoch, "Accuracy : {:.4f}".format(valid_acc))

            if valid_acc > max_acc:
                max_acc = valid_acc

                saver.save(s, model_)

        if epoch % 1 == 0:
            print("[*] Epoch %03d" % epoch, "Cost : {:.10f}".format(avg_cost), "Max Accuracy : {:.4f}".format(max_acc))
            print("---------------------------------------------")

    end_time = time.time() - start_time

    # elapsed time
    print("[+] Elapsed time {:.10f}s".format(end_time))

    s.close()
