import numpy as np

from neural_network_approach.learning_strategies.strategy import LearningStrategy


class Adagrad(LearningStrategy):
    def __init__(self, layers, learning_factor):
        super().__init__(layers, learning_factor)
        self.epsilon = 1e-8
        self.grad_square_sum_list = self.init_layer_tuples_list()

    def update_single_layer(self, layer_id, sum_weights, sum_bias, batch_size):
        self.layers[layer_id].weights -= np.divide(
            self.learning_factor * sum_weights,
            np.sqrt(self.grad_square_sum_list[layer_id][0] + self.epsilon) * batch_size
        )
        self.layers[layer_id].bias_weight -= np.divide(
            self.learning_factor * sum_bias,
            np.sqrt(self.grad_square_sum_list[layer_id][1] + self.epsilon) * batch_size
        )

        new_tuple = (
            self.grad_square_sum_list[layer_id][0] + np.square(sum_weights),
            self.grad_square_sum_list[layer_id][1] + np.square(sum_bias)
        )

        self.grad_square_sum_list[layer_id] = new_tuple
