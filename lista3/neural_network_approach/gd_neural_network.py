import random
import time

import numpy as np

from neural_network_approach.weights_initialization.weights_init_utils import process_weights_inits_string
from neural_network_approach.utils import process_layers
from neural_network_approach.learning_strategies.strategy_utils import process_learning_strategy_string


class GDNeuralNetwork:
    def __init__(self, input_data_size, layers, epochs, mini_batch):
        self.input_data_size = input_data_size
        self.layers_raw = layers
        self.strategy = None
        self.epochs_limit = epochs
        self.mini_batch_size = mini_batch

        self.min_error_layers_states = []
        self.min_error = 1
        self.max_error_difference = 0.01

    def compile(self, weights_init_method, strategy, learning_factor):
        weights_init_class = process_weights_inits_string(weights_init_method)
        layers = process_layers(self.input_data_size, self.layers_raw, weights_init_class)
        strategy_class = process_learning_strategy_string(strategy)
        self.strategy = strategy_class(
            layers=layers,
            learning_factor=learning_factor
        )
        self.layers_raw = None

    def proceed_forward(self, x):
        last_result = x
        for j in range(len(self.strategy.layers)):
            self.strategy.layers[j].calculate_stimulation(last_result)
            self.strategy.layers[j].proceed_activation()
            last_result = self.strategy.layers[j].activation
        return last_result

    def proceed_backward(self, x, y, forward_result):
        weights_updates = []

        # layer last
        last_error, weights_delta, bias_delta = self.calculate_last_layer_error(y, forward_result)
        weights_updates.insert(0, (weights_delta, bias_delta))

        # hidden layers
        for i in range(len(self.strategy.layers) - 2, -1, -1):
            last_error, weights_delta, bias_delta = self.calculate_hidden_layer_error(
                last_error=last_error,
                layer_id=i,
                grad_layer_activation=x if i == 0 else self.strategy.layers[i - 1].activation
            )
            weights_updates.insert(0, (weights_delta, bias_delta))

        return weights_updates

    def calculate_last_layer_error(self, y, forward_result):
        error = 2 * (forward_result - y) / forward_result.shape[0] * self.strategy.layers[-1].activation_function(
            x=self.strategy.layers[-1].stimulation,
            derivative=True
        )
        return error, np.outer(error, self.strategy.layers[-2].activation), np.row_stack(error)

    def calculate_hidden_layer_error(self, last_error, layer_id, grad_layer_activation):
        error = np.dot(self.strategy.layers[layer_id + 1].weights.T, last_error) * self.strategy.layers[layer_id].activation_function(
            x=self.strategy.layers[layer_id].stimulation,
            derivative=True
        )
        return error, np.outer(error, grad_layer_activation), np.row_stack(error)

    def fit(self, train_data, val_data, stochastic=True):
        (x_train, y_train), (x_val, y_val) = train_data, val_data
        train_values = list(zip(x_train.values, y_train))
        epoch = 0
        start_time = time.time()
        time_threshold = 300
        list_epochs = []
        list_errors = []

        while self.epochs_limit > epoch and (time.time() - start_time) < time_threshold:
            iterator = 0

            if stochastic:
                random.shuffle(train_values)

            weights_updates_list = []
            for x, y in train_values:
                weights_updates = self.proceed_backward(x, y, self.proceed_forward(x))
                weights_updates_list.append(weights_updates)
                iterator += 1

                if iterator == self.mini_batch_size:
                    self.strategy.update_weights(weights_updates_list)
                    iterator = 0
                    weights_updates_list = []

            acc = self.evaluate(x_val, y_val)
            print(f"Epoch: {epoch + 1}, acc: {acc}, time: {str(int(time.time() - start_time))}")
            list_epochs.append(epoch + 1)
            list_errors.append(1 - acc)
            self.analyse_error(1 - acc)
            epoch += 1

        return epoch, 1 - self.min_error, list_epochs, list_errors

    def analyse_error(self, error):
        if (error - self.min_error) > self.max_error_difference:
            self.strategy.layers = self.min_error_layers_states
            print("Early stopping!")
        elif error - self.min_error < 0:
            self.min_error = error
            self.min_error_layers_states = self.strategy.layers

    def predict(self, x):
        return self.proceed_forward(x)

    def evaluate(self, x_test, y_test):
        lst = []
        for x, y in zip(x_test.values, y_test):
            lst.append(np.argmax(self.predict(x)) == np.argmax(y))
        return sum(lst) / len(lst)
