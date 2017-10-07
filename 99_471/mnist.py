import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


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


def SCNN():
    model = Sequential()

    filter = 32
    model.add(Conv2D(filters=filter, kernel_size=(5, 5), padding='SAME', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=filter, kernel_size=(5, 5), padding='SAME', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    filter *= 2
    model.add(Conv2D(filters=filter, kernel_size=(3, 3), padding='SAME', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=(3, 3), padding='SAME', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer=RMSprop(lr=1e-3, rho=0.9, epsilon=1e-8, decay=0.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    return model

data_generate = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
)
data_generate.fit(x_train)

epochs = 50
batch_size = 64
model = SCNN()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)

history = model.fit_generator(data_generate.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                              validation_data=(x_valid, y_valid), verbose=2,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction, early_stopping]
                              )

results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("submit.csv", index=False)
