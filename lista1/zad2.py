import numpy as np
import random


class Adaline:

    def __init__(self):
        self.DEBUG = True
        self.weight_range = [-0.1, 0.1]
        self.weights = np.array(0)

        self.bias_weight = random.uniform(self.weight_range[0], self.weight_range[1])
        self.bias_value = 1.0
        self.is_bias = True

        self.threshold = random.random()

        self.learning_factor = 0.01

        self.epochs = 0

        self.error_threshold = 0.1

        self.train_inputs = np.array(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0],
             [-1.00002, -1.001], [-1.003, 1.00101], [1.0112, -1.00201], [1.0006, 1.00006]]
        )
        self.train_outputs = np.array([-1, -1, -1, 1, -1, -1, -1, 1])

        self.generate_weights()

    def generate_weights(self):
        self.weights = np.array([random.uniform(self.weight_range[0], self.weight_range[1]),
                                 random.uniform(self.weight_range[0], self.weight_range[1])])

    def activate_function(self, x):
        if self.is_bias:
            threshold = 0
        else:
            threshold = self.threshold

        if x > threshold:
            return 1
        else:
            return -1

    def calculate_entire_error(self):
        sum = 0.0
        vector_size = len(self.train_inputs)

        for i in range(vector_size):
            propagation_result = self.propagation(self.train_inputs[i], bias=self.is_bias)
            error = float(self.train_outputs[i]) - propagation_result
            sum += error ** 2

        return float(sum) / float(vector_size)

    def update_weights(self, result, offset):
        error = self.train_outputs[offset] - result
        for i in range(len(self.weights)):
            self.weights[i] += 2 * self.learning_factor * error * self.train_inputs[offset][i]
        self.bias_weight += 2 * self.learning_factor * error * self.bias_value

    def pick_index(self, used):
        if len(used) == len(self.train_inputs):
            return False, -1
        index = random.randint(0, len(self.train_inputs) - 1)
        while index in used:
            return self.pick_index(used)
        return True, index

    def train(self):
        while self.calculate_entire_error() > self.error_threshold:
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
                print(f"Error: {self.calculate_entire_error()}")

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
        print(f"Error threshold: {self.error_threshold}")
        print(f"Actual entire error: {self.calculate_entire_error()}")
        print("---------------------------------------------------")
