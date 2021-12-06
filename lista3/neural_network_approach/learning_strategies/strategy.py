class LearningStrategy:
    def __init__(self, layers, learning_factor):
        self.layers = layers
        self.learning_factor = learning_factor

    def init_layer_tuples_list(self):
        return [(0, 0) for i in range(len(self.layers))]

    def update_weights(self, updates_list):
        batch_size = len(updates_list)

        for i in range(len(self.layers)):
            sum_weights = 0.0
            sum_bias = 0.0
            for j in range(batch_size):
                sum_weights += updates_list[j][i][0]
                sum_bias += updates_list[j][i][1]
            self.update_single_layer(i, sum_weights, sum_bias, batch_size)

    def update_single_layer(self, layer_id, sum_weights, sum_bias, batch_size):
        self.layers[layer_id].weights -= self.learning_factor * sum_weights / batch_size
        self.layers[layer_id].bias_weight -= self.learning_factor * sum_bias / batch_size
