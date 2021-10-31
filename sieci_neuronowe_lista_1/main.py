from zad1 import SimpleNeuralNetwork
import numpy as np
import warnings

from zad2 import Adaline


def start_zad_1():
    print("Hello")

    network = SimpleNeuralNetwork()
    print(network.weights)

    network.train()

    print(network.weights)

    print("Testing the data")
    test_data = np.array([[0.0001, 0.00201], [0.002, 0.999], [1.022, 0.002], [1, 1], ])

    for data in test_data:
        print(f"Result for {data} is:")
        print(network.propagation(data))


def start_zad_2():
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
    start_zad_2()
