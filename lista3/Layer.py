import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1 - (np.tanh(x) ** 2)
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return np.maximum(x, 0).astype(int)
    return np.maximum(x, 0)


def softmax(x, derivative=False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


class Layer:
    def __init__(self, input_size, layer_size, activation_function):
        self.input_size = input_size
        self.layer_size = layer_size
        self.weights = []
        self.weights_predicted = []
        self.momentum_saved_weights = []
        self.weights_increase = []
        self.weights_update = []
        self.activation_function = activation_function
        self.stimulation = np.zeros(layer_size)
        self.activation = np.zeros(layer_size)
        self.error = 0
        self.gradients_sum = []

        self.bias = 1
        self.bias_weight = np.zeros(layer_size)

        self.generate_weights()

    def generate_weights(self):
        scale = 0.01
        self.weights = np.random.normal(scale=scale, size=(self.layer_size, self.input_size))
        self.bias_weight = np.random.normal(scale=scale, size=(self.layer_size, 1))
        self.weights_increase = np.zeros(shape=(self.layer_size, self.input_size))
        self.weights_predicted = self.weights
        self.gradients_sum = np.zeros(shape=(self.layer_size, self.input_size))

    def calculate_stimulation(self, x, predicted=False):
        a = np.insert(x, 0, self.bias)
        w = np.hstack((self.bias_weight, self.get_weights(predicted)))
        self.stimulation = np.dot(w, a)

    def proceed_activation(self):
        self.activation = self.activation_function(self.stimulation)

    def get_weights(self, predicted=False):
        if predicted:
            return self.weights_predicted
        else:
            return self.weights
