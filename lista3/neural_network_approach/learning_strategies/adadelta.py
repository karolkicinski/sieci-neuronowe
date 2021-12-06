import numpy as np

from neural_network_approach.learning_strategies.strategy import LearningStrategy


class Adadelta(LearningStrategy):
    def __init__(self, layers, learning_factor):
        super().__init__(layers, learning_factor)
        self.gamma = 0.9
        self.epsilon = 1e-8
        self.last_exp_avg_grads = self.init_layer_tuples_list()
        self.last_exp_avg_deltas = self.init_layer_tuples_list()
        self.last_deltas = self.init_layer_tuples_list()

    def update_single_layer(self, layer_id, sum_weights, sum_bias, batch_size):
        rms_grad, rms_grad_bias = self.calculate_rms_grad(layer_id, sum_weights, sum_bias)
        rms_delta, rms_delta_bias = self.calculate_rms_delta(layer_id)

        deltas_update_weights = np.divide(-rms_delta * sum_weights, rms_grad)
        deltas_update_bias = np.divide(-rms_delta_bias * sum_bias, rms_grad_bias)
        self.last_deltas[layer_id] = (deltas_update_weights, deltas_update_bias)

        self.layers[layer_id].weights += deltas_update_weights / batch_size
        self.layers[layer_id].bias_weight += deltas_update_bias / batch_size

    def calculate_rms_grad(self, layer_id, sum_weights, sum_bias):
        weights_update = self.gamma * self.last_exp_avg_grads[layer_id][0] + (1 - self.gamma) * np.square(sum_weights)
        bias_update = self.gamma * self.last_exp_avg_grads[layer_id][1] + (1 - self.gamma) * np.square(sum_bias)
        self.last_exp_avg_grads[layer_id] = (weights_update, bias_update)
        return np.sqrt(weights_update + self.epsilon), np.sqrt(bias_update + self.epsilon)

    def calculate_rms_delta(self, layer_id):
        weights_update = self.gamma * self.last_exp_avg_deltas[layer_id][0] + (1 - self.gamma) * np.square(self.last_deltas[layer_id][0])
        bias_update = self.gamma * self.last_exp_avg_deltas[layer_id][1] + (1 - self.gamma) * np.square(self.last_deltas[layer_id][1])
        self.last_exp_avg_deltas[layer_id] = (weights_update, bias_update)
        return np.sqrt(weights_update + self.epsilon), np.sqrt(bias_update + self.epsilon)


