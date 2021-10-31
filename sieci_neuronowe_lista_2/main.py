import numpy as np

from etap1 import NeuralNetwork


def start():
    print("Start!")

    patterns = np.array([
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1 ])
    ])

    network = NeuralNetwork(patterns)
    network.generate_layers()
    network.proceed()

    print("End!")


if __name__ == '__main__':
    start()

