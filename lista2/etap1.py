import math

import numpy as np
from scipy.special import expit, softmax

from Layer import Layer


class NeuralNetwork:

    def __init__(self, learning_set):
        self.learning_set = learning_set

        self.number_of_layers = 2
        self.layers = []

    def generate_layers(self):
        # layer 1
        layer_1_size = 5
        self.layers.append(Layer(layer_1_size, expit, len(self.learning_set[0][0])))

        # layer 2
        layer_2_size = 9
        self.layers.append(Layer(layer_2_size, math.tanh, layer_1_size))

        # layer 3 - last
        layer_3_size = len(self.learning_set[0][0])
        self.layers.append(Layer(layer_3_size, math.tanh, layer_2_size))

    def proceed(self):
        for i in range(len(self.learning_set)):
            last_x = self.learning_set[i][0]
            last_y = self.learning_set[i][1]

            for j in range(len(self.layers)):
                self.layers[j].calculate_stimulation(last_x)

                if j is not len(self.layers) - 1:
                    self.layers[j].proceed_activation()
                    last_x = self.layers[j].activation
                else:
                    result = softmax(self.layers[j].stimulation)
                    print(result)
                    print(f"Total: {sum(result)}")



