import tensorflow as tf


def conv2d(x, w, stride_=1):
    return tf.nn.conv2d(x, w, strides=[1, stride_, stride_, 1], padding='SAME')


def max_pool(x, kernel_=2, stride_=2):
    # return tf.nn.max_pool(x, ksize=[1, kernel_, kernel_, 1], strides=[1, stride_, stride_, 1], padding='SAME')
    return tf.layers.max_pooling2d(x, pool_size=[kernel_, kernel_], strides=stride_)


def cnn_model_0(x,  # 28 x 28 x 1 images
                w,  # Weight parameters
                b,  # bias parameters
                p_keep_conv_=0.8,  # dropout rate
                p_keep_hidden_=0.5,
                ):

    with tf.name_scope("cnn_layer_0"):
        conv0 = tf.nn.bias_add(tf.nn.relu(conv2d(x, w[0])), b[0])
        pool0 = max_pool(conv0)
        layer0 = tf.nn.dropout(pool0, p_keep_conv_)

    with tf.name_scope("cnn_layer_1"):
        conv1 = tf.nn.bias_add(tf.nn.relu(conv2d(layer0, w[1])), b[1])
        layer1 = tf.nn.dropout(conv1, p_keep_conv_ - 0.05)

    with tf.name_scope("cnn_layer_2"):
        conv3 = tf.nn.bias_add(tf.nn.relu(conv2d(layer1, w[2], 2)), b[2])
        layer3 = tf.reshape(conv3, [-1, w[3].get_shape().as_list()[0]])
        layer3 = tf.nn.dropout(layer3, p_keep_conv_ - 0.1)

    with tf.name_scope("cnn_layer_3"):
        conv4 = tf.add(tf.nn.relu(tf.matmul(layer3, w[3])), b[3])
        layer4 = tf.nn.dropout(conv4, p_keep_hidden_)

    with tf.name_scope("hypothesis"):  # 1024
        _X = tf.add(tf.matmul(layer4, w[4]), b[4])
        _X = tf.contrib.layers.flatten(_X)

    return _X
