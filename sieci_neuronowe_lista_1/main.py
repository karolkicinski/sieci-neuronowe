from zad1 import SimpleNeuralNetwork
import numpy as np
import warnings

from zad2 import Adaline


def lab1_test():
    print("Hello")

    network = SimpleNeuralNetwork()
    print(network.weights)

    network.train()

    print(network.weights)

    print("Testing the data")
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], ])

    for data in test_data:
        print(f"Result for {data} is:")
        print(network.propagation(data))


def lab1_report():
    print("==========> ZAD. 1")

    theta = [0.1, 0.3, 0.5, 0.9, 1, 1.1, 2, 5, 10, 20, 50, 100]

    for t in theta:
        epochs = []
        for i in range(10):
            network = SimpleNeuralNetwork()
            network.DEBUG = False
            network.is_bias = False
            network.threshold = t
            network.train()
            epochs.append(network.epochs)

        print(f"Theta = {t}, AV EPOCHS: {sum(epochs) / len(epochs)}")

    # ==================================================================

    print("==========> ZAD. 2")

    weight_ranges = [[-1.0, 1.0], [-0.8, 0.8], [-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3],
                     [-0.2, 0.2], [-0.1, 0.1], [-0.01, 0.01], [-0.001, 0.001], [-0.0001, 0.0001]]

    for w in weight_ranges:
        epochs = []
        for i in range(10):
            network = SimpleNeuralNetwork()
            network.DEBUG = False
            network.is_bias = True
            network.weight_range = w
            network.generate_weights()
            network.train()
            epochs.append(network.epochs)

        print(f"Weight range = {w}, AV EPOCHS: {sum(epochs) / len(epochs)}")

    # ==================================================================

    print("==========> ZAD. 3")

    alpha = [10, 5, 1.0, 0.5, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001]

    for a in alpha:
        epochs = []
        for i in range(10):
            network = SimpleNeuralNetwork()
            network.DEBUG = False
            network.is_bias = False
            network.learning_factor = a
            network.train()
            epochs.append(network.epochs)

        print(f"Alpha = {a}, AV EPOCHS: {sum(epochs) / len(epochs)}")

    # ==================================================================

    print("==========> ZAD. 4")

    epochs = []
    for i in range(10):
        network = SimpleNeuralNetwork()
        network.DEBUG = False
        network.is_bias = False
        network.train()
        epochs.append(network.epochs)

    print(f"AV EPOCHS: {sum(epochs) / len(epochs)}")


def lab2_test():
    print("Hello in Adaline!\n")

    network = Adaline()
    print(f"Activation threshold: {network.threshold_bi}")
    print(f"Error threshold: {network.error_threshold}")
    print(f"Zero weight: \n {network.zero_weight}")
    print(f"Weights: \n {network.weights}")

    network.train()

    print(f"Iterations: {network.iterations}")
    print(f"Activation threshold: {network.threshold_bi}")
    print(f"Error threshold: {network.error_threshold}")
    print(f"Zero weight: \n {network.zero_weight}")
    print(f"Weights: \n {network.weights}")

    print("\n====> Testing <====\n")
    test_data = np.array([[-1.0001, -1.00201], [-1.002, 0.999], [1.022, -1.002], [1, 1], ])

    for data in test_data:

        print(f"Result for {data} is: {network.propagation(data)}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    lab2_test()
    # lab1_report()
