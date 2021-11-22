import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from etap1 import NeuralNetwork
import pickle

FILE = "mnist.txt"

def start():
    print("Start!")

    # x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # save_mnist(x, y)
    x, y = load_mnist()
    x = (x / 255).astype('float32')
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.0425, test_size=0.0075, random_state=42)

    network = NeuralNetwork()
    network.generate_layers()
    epoch, acc = network.train(x_train, x_val, y_train, y_val)
    print(f"Epochs: {epoch}, acc: {acc}!")

    print("End!")


def badania():
    print("Start!")

    # x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # save_mnist(x, y)
    x, y = load_mnist()
    x = (x / 255).astype('float32')
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.0425, test_size=0.0075, random_state=42)

    results_ep = []
    results_acc = []

    replies = 10

    for i in range(replies):
        network = NeuralNetwork()
        network.generate_layers()
        epoch, acc = network.train(x_train, x_val, y_train, y_val)
        results_ep.append(epoch)
        results_acc.append(acc)

    print(f"RESULT: av epochs: {sum(results_ep)/len(results_ep)}, av acc: {sum(results_acc)/len(results_acc)}")

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

