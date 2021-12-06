import numpy as np


class WeightsInitializer:
    def __init__(self, sizes):
        self.input_size, self.layer_size = sizes
        self.standard_scale = 0.1

    def generate_weights_custom_scale(self, scale):
        weights = np.random.normal(scale=scale, size=(self.layer_size, self.input_size))
        bias_weight = np.random.normal(scale=scale, size=(self.layer_size, 1))
        return weights, bias_weight

    def generate_weights(self):
        return self.generate_weights_custom_scale(scale=self.standard_scale)
