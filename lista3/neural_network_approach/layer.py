import numpy as np

from neural_network_approach.activation.activation_utils import process_activation_function_string
from neural_network_approach.weights_initialization.weights_initializer import WeightsInitializer


class Layer:
    def __init__(self, layer_size, activation_function):
        self.weights = None
        self.bias_weight = None
        self.bias = 1
        self.input_size = None
        self.layer_size = layer_size

        self.activation_function = process_activation_function_string(activation_function)
        self.weights_init_approach: WeightsInitializer = None

        self.stimulation = None
        self.activation = None

    def init(self):
        self.generate_weights()

    def generate_weights(self):
        self.weights, self.bias_weight = self.weights_init_approach.generate_weights()

    def calculate_stimulation(self, x):
        a = np.insert(x, 0, self.bias)
        w = np.hstack((self.bias_weight, self.weights))
        self.stimulation = np.dot(w, a)

    def proceed_activation(self):
        self.activation = self.activation_function(self.stimulation)
