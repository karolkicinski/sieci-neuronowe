from neural_network_approach.learning_strategies.momentum import Momentum


class MomentumNesterov(Momentum):
    def __init__(self, layers, learning_factor):
        super().__init__(layers, learning_factor)
        self.actual_weights = self.init_actual_weights_list()

    def init_actual_weights_list(self):
        return [(layer.weights, layer.bias_weight) for layer in self.layers]

    def update_single_layer(self, layer_id, sum_weights, sum_bias, batch_size):
        weights_update = self.momentum_factor * self.weights_increase_list[layer_id][0] + self.learning_factor * sum_weights / batch_size
        bias_weights_update = self.momentum_factor * self.weights_increase_list[layer_id][1] + self.learning_factor * sum_bias / batch_size

        self.weights_increase_list[layer_id] = (weights_update, bias_weights_update)

        self.layers[layer_id].weights = self.actual_weights[layer_id][0] - weights_update
        self.layers[layer_id].bias_weight = self.actual_weights[layer_id][1] - bias_weights_update

        self.actual_weights[layer_id] = (self.layers[layer_id].weights, self.layers[layer_id].bias_weight)

        self.layers[layer_id].weights -= self.momentum_factor * weights_update
        self.layers[layer_id].bias_weight -= self.momentum_factor * bias_weights_update
