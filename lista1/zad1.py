import math

import numpy as np
import random


class SimpleNeuralNetwork:

    def __init__(self):
        self.DEBUG = True
        self.weight_range = [-0.1, 0.1]
        self.weights = np.array(0)

        self.bias_weight = random.uniform(self.weight_range[0], self.weight_range[1])
        self.bias_value = 1.0
        self.is_bias = True

        self.threshold_uni = random.random()
        self.threshold_bi = random.uniform(-0.1, 0.1)

        self.threshold = self.threshold_uni
        self.activate_function = self.activate_function_uni

        self.learning_factor = 0.01
        self.is_error = 1

        self.epochs = 0

        self.train_inputs = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
             [0.00002, 0.001], [0.003, 1.00101], [1.0112, 0.00201], [1.0006, 1.00006]]
        )
        self.train_outputs = np.array([0, 0, 0, 1, 0, 0, 0, 1])
        # self.train_inputs = np.array(
        #     [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0],
        #      [-1.00002, -1.001], [-1.003, 1.00101], [1.0112, -1.00201], [1.0006, 1.00006]]
        # )
        # self.train_outputs = np.array([-1, -1, -1, 1, -1, -1, -1, 1])

        self.generate_weights()

    def generate_weights(self):
        self.weights = np.array([random.uniform(self.weight_range[0], self.weight_range[1]),
                                 random.uniform(self.weight_range[0], self.weight_range[1])])

    def activate_function_uni(self, x):
        if self.is_bias:
            threshold = 0
        else:
            threshold = self.threshold

        if x > threshold:
            return 1
        else:
            return 0

    def activate_function_bi(self, x):
        if self.is_bias:
            threshold = 0
        else:
            threshold = self.threshold

        if x > threshold:
            return 1
        else:
            return -1

    def update_weights(self, result, offset):
        error = self.train_outputs[offset] - result
        for i in range(len(self.weights)):
            self.weights[i] += error * self.train_inputs[offset][i] * self.learning_factor
        self.bias_weight += error * self.bias_value * self.learning_factor
        self.is_error += math.fabs(error)

    def pick_index(self, used):
        if len(used) == len(self.train_inputs):
            return False, -1
        index = random.randint(0, len(self.train_inputs) - 1)
        while index in used:
            return self.pick_index(used)
        return True, index

    def train(self):
        while self.is_error != 0:
            self.is_error = 0
            self.epochs += 1
            used_indexes = []
            result, index = self.pick_index(used_indexes)

            if self.epochs >= 10000:
                print("Epochs out of range 10000!")
                break

            while result:
                propagation_result = self.propagation(self.train_inputs[index], self.is_bias)
                self.update_weights(propagation_result, index)

                used_indexes.append(index)
                result, index = self.pick_index(used_indexes)

        if self.DEBUG:
            print("end of training")
            print(f"epochs: {self.epochs}")

    def propagation(self, inputs, bias=False):
        if bias:
            w = np.insert(self.weights, 0, self.bias_weight)
            x = np.insert(inputs, 0, self.bias_value)
        else:
            w = self.weights
            x = inputs
        return self.activate_function(np.dot(x, w))

    def print_result(self):
        print("---------------------------------------------------")
        print(f"Weights: {self.weights}")
        print(f"Bias weight: {self.bias_weight}")
        print(f"Is bias mode: {self.is_bias}")
        print(f"Epochs: {self.epochs}")
        print(f"Activate function: {self.activate_function.__name__}")
        print(f"Threshold: {self.threshold}")
        print("---------------------------------------------------")
