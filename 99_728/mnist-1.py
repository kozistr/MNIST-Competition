import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

import keras as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def SELU(x):
    # fixed point mean, var (0, 1)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return scale * K.elu(x, alpha)


# reproducibility
seed = 777
tf.set_random_seed(seed)
np.random.seed(seed)


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

y_train = train['label']
y_train = to_categorical(y_train, num_classes=10)
x_train = train.drop(labels=['label'], axis=1)

del train

# normalize
x_train /= 255.
test /= 255.

# reshape
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)


def vgg():
    def two_conv_pool(x, F1, F2, name, input_):
        if input_:
            x.add(Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name),
                         input_shape=(28, 28, 1)))
        else:
            x.add(Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name)))

        x.add(BatchNormalization())
        x.add(Activation('relu'))

        x.add(Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name)))
        x.add(BatchNormalization())
        x.add(Activation('relu'))

        x.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='{}_pool'.format(name)))

        return x

    def three_conv_pool(x, F1, F2, F3, name):
        x.add(Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name)))
        x.add(BatchNormalization())
        x.add(Activation('relu'))

        x.add(Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name)))
        x.add(BatchNormalization())
        x.add(Activation('relu'))

        x.add(Conv2D(F3, (3, 3), activation=None, padding='same', name='{}_conv3'.format(name)))
        x.add(BatchNormalization())
        x.add(Activation('relu'))

        x.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='{}_pool'.format(name)))

        return x

    net = Sequential()

    net = two_conv_pool(net, 64, 64, "block1", True)
    net = two_conv_pool(net, 128, 128, "block2", False)
    net = three_conv_pool(net, 256, 256, 256, "block3")
    net = three_conv_pool(net, 512, 512, 512, "block4")

    net.add(Flatten())
    net.add(Dense(512, activation='relu', name='fc'))
    net.add(Dense(10, activation='softmax', name='predictions'))

    net.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    net.summary()

    return net

data_generate = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    horizontal_flip=False,
    vertical_flip=False,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
)
data_generate.fit(x_train)

epochs = 600
batch_size = 64

model = vgg()

# tensor_board = TensorBoard(log_dir='./logs/', histogram_freq=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=1e-5)

model.fit_generator(data_generate.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                    validation_data=(x_valid, y_valid), verbose=2,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction, early_stopping,
                               ModelCheckpoint('./mnist-cnn-1.ckpt', monitor='val_acc', save_best_only=True)]
                    )

# model.save_weights('./mnist-cnn-1.h5')

model.load_weights('./mnist-cnn-1.ckpt')

results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("submit-1.csv", index=False)
