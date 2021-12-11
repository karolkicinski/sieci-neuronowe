import csv

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
import matplotlib.pyplot as plt

import NeuralNetwork
from tensorflow import keras

FILE = "mnist.txt"


def badania():
    print("Start!")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    epochs = 5

    parameter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iters = 10
    list_epochs = []
    list_errors = []
    final_list_acc = []
    final_list_param = []

    with open('7.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pooling size', 'epochs', 'accuracy'])

        for p in parameter:
            print(f"====================> PARAM VALUE: {p}")
            results_acc = []
            results_ep = []

            for i in range(iters):
                model = NeuralNetwork.generate_cnn(layers=1, filter_size=3, poolings=1, pool_size=p)
                # model = NeuralNetwork.generate_fully_connected(layers=p)
                model.fit(x_train, y_train, epochs=epochs)
                acc = model.evaluate(x_test, y_test, verbose=2)

                results_acc.append(acc[1])
                results_ep.append(epochs)

            final_list_acc.append(sum(results_acc)/len(results_acc))
            final_list_param.append(p)
            writer.writerow([p, round(sum(results_ep)/len(results_ep)), round(sum(results_acc)/len(results_acc), 4)])

        plt.plot(final_list_param, final_list_acc)
        plt.ylabel('Accuracy')
        plt.xlabel('Pooling size')
        plt.title(f"Badanie wpływu wielkości filtra dla pooling na szybkość uczenia sieci.")
        plt.show()


def start():
    print("Start!")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    epochs = 3
    model = NeuralNetwork.generate_cnn()
    # model = NeuralNetwork.generate_fully_connected()
    model.fit(x_train, y_train, epochs=epochs)
    acc = model.evaluate(x_test, y_test, verbose=2)

    print(f"Epochs: {epochs}, acc: {round(acc[1] * 100, 2)}%!")

    print("End!")


def pl():
    plt.plot(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0.9637, 0.9754, 0.9807, 0.9843, 0.9844, 0.9856, 0.9855, 0.9867, 0.9869, 0.9871]
    )
    plt.ylabel('Accuracy')
    plt.xlabel('Layers')
    plt.title(f"Badanie wpływu ilości warstw na szybkość uczenia sieci.")
    plt.show()



if __name__ == '__main__':
    # start()
    badania()
    # pl()