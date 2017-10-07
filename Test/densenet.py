import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import metrics
from sklearn import model_selection

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 1, 1,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4):
    for i in range(nb_layers):
        merge_tensor = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([merge_tensor, x], axis=-1)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes=10, nb_dense_block=4, growth_rate=48,
             nb_filter=96, dropout_rate=None, weight_decay=1e-4):

    net_in = Input(shape=(28, 28, 1))

    nb_layers = [6, 12, 24, 16]  # dense-net 161

    # Initial convolution
    net = Convolution2D(nb_filter, 3, 3,
                        init="he_uniform",
                        border_mode="same",
                        name="initial_conv2D",
                        bias=False,
                        W_regularizer=l2(weight_decay))(net_in)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        net, nb_filter = denseblock(net, nb_layers[block_idx], nb_filter, growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition
        net = transition(net, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock does not have a transition
    net, nb_filter = denseblock(net, nb_layers[-1], nb_filter, growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

    net = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(net)

    net = Activation('relu')(net)
    net = GlobalAveragePooling2D()(net)
    net = Dense(nb_classes,
                activation='softmax',
                W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(net)

    net = Model(input=net_in, output=net, name="Wide-DenseNet-BC")

    net.compile(optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                loss="categorical_crossentropy", metrics=["accuracy"])

    net.summary()

    return net