import time

import numpy as np

import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.learning_factor = 2
        self.epochs_limit = 800
        self.batch_size = 16

        self.min_error_layers_states = []
        self.min_error = 1
        self.max_error_difference = 0.01

    def generate_layers(self):
        layers_sizes = [784, 10, 10]
        layers_func = [Layer.relu, Layer.softmax]

        for i in range(len(layers_func)):
            self.layers.append(Layer.Layer(layers_sizes[i], layers_sizes[i+1], layers_func[i]))

    def proceed_forward(self, x):
        last_result = x

        for j in range(len(self.layers)):
            self.layers[j].calculate_stimulation(last_result)
            self.layers[j].proceed_activation()
            last_result = self.layers[j].activation

        # print(last_result)
        # print(f"Total: {sum(last_result)}")
        return last_result

    def proceed_backward(self, x, y, forward_result):
        return_update = []

        # layer last
        last_error = 2 * (forward_result - y) / forward_result.shape[0] * self.layers[-1].activation_function(self.layers[-1].stimulation, True)
        return_update.insert(0, np.outer(last_error, self.layers[-2].activation))

        l = len(self.layers) - 2
        for i in range(l):
            lay_id = -1 - i
            last_error = np.dot(self.layers[lay_id].weights.T, last_error) * self.layers[lay_id - 1].activation_function(self.layers[lay_id - 1].stimulation, True)
            return_update.insert(0, np.outer(last_error, self.layers[lay_id - 2].activation))

        # layer first
        last_error = np.dot(self.layers[1].weights.T, last_error) * self.layers[0].activation_function(self.layers[0].stimulation, True)
        return_update.insert(0, np.outer(last_error, x))

        return return_update

    # Code to train without batch

    # def train(self, x_train, x_val, y_train, y_val):
    #     epoch = 0
    #     while self.epochs_limit > epoch:
    #         for x, y in zip(x_train.values, y_train):
    #             self.update_weights(self.proceed_backward(x, y, self.proceed_forward(x)))
    #
    #         print(f"Epoch: {epoch}, acc: {self.check_fit(x_val, y_val)}")
    #
    # def update_weights(self, update_grads):
    #     for i in range(len(self.layers)):
    #         self.layers[i].weights = self.layers[i].weights - self.learning_factor * update_grads[i]

    def train(self, x_train, x_val, y_train, y_val):
        epoch = 0
        start_time = time.time()
        time_threshold = 60
        acc = -1
        while self.epochs_limit > epoch and (time.time() - start_time) < time_threshold:
            iterator = 0
            update_grads = []
            for x, y in zip(x_train.values, y_train):
                new_update_grads = self.proceed_backward(x, y, self.proceed_forward(x))
                if iterator < self.batch_size:
                    iterator += 1
                    update_grads.append(new_update_grads)
                else:
                    self.update_weights(update_grads)
                    iterator = 0
                    update_grads = []

            acc = self.check_fit(x_val, y_val)
            print(f"Epoch: {epoch + 1}, acc: {acc}, time: {str(int(time.time() - start_time))}")
            self.analyse_error(1 - acc)
            epoch += 1

        return epoch + 1, 1 - self.min_error

    def analyse_error(self, error):
        if (error - self.min_error) > self.max_error_difference:
            self.layers = self.min_error_layers_states
            print("Early stopping!")
        elif error - self.min_error < 0:
            self.min_error = error
            self.min_error_layers_states = self.layers

    def update_weights(self, update_grads):
        for i in range(len(self.layers)):
            suma = 0.0
            for j in range(len(update_grads)):
                suma += update_grads[j][i]
            self.layers[i].weights -= self.learning_factor * suma / len(update_grads)

    def check_fit(self, x_val, y_val):
        lst = []
        for x, y in zip(x_val.values, y_val):
            lst.append(np.argmax(self.proceed_forward(x)) == np.argmax(y))
        return sum(lst) / len(lst)


