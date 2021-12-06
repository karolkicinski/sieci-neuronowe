import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
import matplotlib.pyplot as plt

import NeuralNetwork
from tensorflow import keras

FILE = "mnist.txt"


def start():
    print("Start!")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    epochs = 3
    model = NeuralNetwork.generate_cnn()
    # model = NeuralNetwork.generate_fully_connected()
    model.fit(x_train, y_train, epochs=epochs)
    acc = model.evaluate(x_test, y_test, verbose=2)

    print(f"Epochs: {epochs}, acc: {acc}!")

    print("End!")


def save_mnist(x, y):
    with open(FILE, mode="wb") as file:
        pickle.dump((x, y), file)


def load_mnist():
    with open(FILE, mode="rb") as file:
        (x, y) = pickle.load(file)
        return x, y


if __name__ == '__main__':
    start()
