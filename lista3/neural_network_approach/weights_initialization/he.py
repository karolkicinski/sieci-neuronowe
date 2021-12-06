import numpy as np

from neural_network_approach.weights_initialization.weights_initializer import WeightsInitializer


class He(WeightsInitializer):
    def __init__(self, sizes):
        super().__init__(sizes)

    def generate_scale(self):
        return np.sqrt(2 / self.input_size)

    def generate_weights(self):
        return super().generate_weights_custom_scale(scale=self.generate_scale())

