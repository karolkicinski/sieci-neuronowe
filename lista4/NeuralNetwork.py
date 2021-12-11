import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt


def generate_fully_connected(layers):

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))

    for l in range(layers):
        model.add(Dense(32, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def generate_cnn(layers, filter_size, poolings, pool_size):
    filter_size = filter_size
    pool_size = pool_size

    model = Sequential()

    model.add(Conv2D(32, kernel_size=filter_size, activation='relu', input_shape=(28, 28, 1)))

    for l in range(layers - 1):
        model.add(Conv2D(16, kernel_size=filter_size, activation='relu'))

    for p in range(poolings):
        model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

