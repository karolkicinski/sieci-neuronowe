import numpy as np


class Layer:
    def __init__(self, elements, activation_function, weights_size):
        self.elements = elements
        self.weights_size = weights_size
        self.weights = []
        self.activation_function = activation_function
        self.stimulation = np.zeros(elements)
        self.activation = np.zeros(elements)

        self.bias = 1

        self.generate_weights()

    def generate_weights(self):
        for i in range(self.elements):
            self.weights.append(np.random.normal(scale=1.0, size=self.weights_size + 1))
            # self.weights.append(np.random.uniform(low=-1, high=1, size=self.weights_size+1))

    def calculate_stimulation(self, x):
        a = np.insert(x, 0, self.bias)
        for i in range(len(self.stimulation)):
            self.stimulation[i] = np.dot(a, self.weights[i])

    def proceed_activation(self):
        for i in range(len(self.stimulation)):
            self.activation[i] = self.activation_function(self.stimulation[i])
