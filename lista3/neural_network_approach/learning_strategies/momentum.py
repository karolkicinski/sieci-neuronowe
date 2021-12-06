from neural_network_approach.learning_strategies.strategy import LearningStrategy


class Momentum(LearningStrategy):
    def __init__(self, layers, learning_factor):
        super().__init__(layers, learning_factor)
        self.momentum_factor = 0.7
        self.weights_increase_list = self.init_layer_tuples_list()

    def update_single_layer(self, layer_id, sum_weights, sum_bias, batch_size):
        weights_update = self.momentum_factor * self.weights_increase_list[layer_id][0] + self.learning_factor * sum_weights / batch_size
        bias_weights_update = self.momentum_factor * self.weights_increase_list[layer_id][1] + self.learning_factor * sum_bias / batch_size
        self.weights_increase_list[layer_id] = (weights_update, bias_weights_update)

        self.layers[layer_id].weights -= weights_update
        self.layers[layer_id].bias_weight -= bias_weights_update
