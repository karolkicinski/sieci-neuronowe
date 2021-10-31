import numpy as np
import random


class SimpleNeuralNetwork:

    def __init__(self):
        self.weights = np.array([[random.uniform(-1.0, 1.0)/10, random.uniform(-1.0, 1.0)/10]]).T
        self.threshold_uni = random.random()
        self.threshold_bi = random.uniform(-1.0, 1.0)

        self.learning_factor = 0.1
        self.train_inputs = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1],
             [0.0002, 0.00001], [0.003, 1.00101], [1.0112, 0.00201], [1.0006, 1.00006]]
        )
        self.train_outputs = np.array([0, 0, 0, 1, 0, 0, 0, 1])
        self.test_values = np.array([[0, 0], [0, 1], [1, 0], [1, 1], ])

    def activate_function_uni(self, x):
        output = np.empty(len(x))
        for i in range(len(x)):
            if x[i] > self.threshold_uni:
                output[i] = 1
            else:
                output[i] = 0
        return output

    def activate_function_bi(self, x):
        output = np.empty(len(x))
        for i in range(len(x)):
            if x[i] > self.threshold_bi:
                output[i] = 1
            else:
                output[i] = -1
        return output

    def update_weights(self, result, offset):
        output = np.empty(len(result))
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (self.train_outputs[offset] - result[0]) * self.train_inputs[offset][i] * self.learning_factor
        return output

    def train(self):
        while not self.check_values():
            for i in range(len(self.train_inputs)):
                propagation_result = self.propagation(self.train_inputs[i])
                self.update_weights(propagation_result, i)

        print("end of training")

    def propagation(self, inputs):
        return self.activate_function_uni(np.dot(inputs.astype(float), self.weights))

    def check_values(self):
        for i in range(len(self.train_inputs)):
            propagation_result = self.propagation(self.train_inputs[i])

            if propagation_result[0] != self.train_outputs[i]:
                return False
        return True
