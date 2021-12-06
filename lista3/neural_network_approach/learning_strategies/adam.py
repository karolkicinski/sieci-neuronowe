import numpy as np

from neural_network_approach.learning_strategies.strategy import LearningStrategy


class Adam(LearningStrategy):
    def __init__(self, layers, learning_factor):
        super().__init__(layers, learning_factor)
        self.gamma = 0.9
        self.epsilon = 1e-8
        self.beta = 0.9
        self.exp_weighted_avg = self.init_layer_tuples_list()
        self.exp_weighted_avg_squared = self.init_layer_tuples_list()

    def update_single_layer(self, layer_id, sum_weights, sum_bias, batch_size):
        v_weights = self.beta * self.exp_weighted_avg[layer_id][0] + (1 - self.beta) * sum_weights
        v_bias = self.beta * self.exp_weighted_avg[layer_id][1] + (1 - self.beta) * sum_bias
        self.exp_weighted_avg[layer_id] = (v_weights, v_bias)
        v_weights = np.divide(v_weights, 1 - self.beta)
        v_bias = np.divide(v_bias, 1 - self.beta)

        v_weights_squared = self.gamma * self.exp_weighted_avg_squared[layer_id][0] + (1 - self.gamma) * np.square(sum_weights)
        v_bias_squared = self.gamma * self.exp_weighted_avg_squared[layer_id][1] + (1 - self.gamma) * np.square(sum_bias)
        self.exp_weighted_avg_squared[layer_id] = (v_weights_squared, v_bias_squared)
        v_weights_squared = np.divide(v_weights_squared, 1 - self.gamma)
        v_bias_squared = np.divide(v_bias_squared, 1 - self.gamma)

        self.layers[layer_id].weights -= np.divide(
            self.learning_factor * v_weights,
            np.sqrt(v_weights_squared) + self.epsilon
        ) / batch_size
        self.layers[layer_id].bias_weight -= np.divide(
            self.learning_factor * v_bias,
            np.sqrt(v_bias_squared) + self.epsilon
        ) / batch_size

