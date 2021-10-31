import numpy as np
import random


class Adaline:

    def __init__(self):
        self.weights = np.array([[random.uniform(-1.0, 1.0) / 10, random.uniform(-1.0, 1.0) / 10]]).T
        self.error_threshold = 0.4
        self.zero_weight = random.uniform(-1.0, 1.0)
        self.bias = 1

        self.learning_factor = 0.01
        self.patterns = np.array([
            ([-1, -1], -1),
            ([-1, 1], -1),
            ([1, -1], -1),
            ([1, 1], 1),
            ([-0.994, -1.01], -1),
            ([-1.034, 0.976], -1),
            ([1.012, -1.026], -1),
            ([1.023, 0.9932], 1),
            ([1.09, 1.020], 1),
            ([0.999, 1.023], 1),
        ])

        self.validation = np.array([
            ([-1, -1], -1),
            ([-1, 1], -1),
            ([1, -1], -1),
            ([1, 1], 1),
            ([-0.999, -0.99999], -1),
            ([-1.000987, 1], -1),
            ([1.00318, -1.00026], -1),
            ([0.999932, 0.99765], 1),
            ([1.0045, 1.0044], 1),
            ([1, 1.0098], 1),
        ]) # zmienic hiperparametry jesli nue ok

        # self.threshold_bi = random.uniform(0, 0.8)
        self.threshold_bi = 0.6
        self.iterations = 0

    def calculate_error(self, offset):
        result = np.dot(self.patterns[offset][0], self.weights)
        return self.patterns[offset][1] - result[0]

    def calculate_entire_error(self):
        return self.calculate_entire_error_for_weights(self.weights)

    def calculate_entire_error_for_weights(self, weights):
        sum = 0.0
        vector_size = len(self.patterns)

        for i in range(vector_size):
            sum += (float(self.patterns[i][1]) - np.dot(self.patterns[i][0], weights)[0]) ** 2

        return sum / vector_size

    def update_weights(self, offset):
        error = self.calculate_error(offset)
        new_weights = np.random.random((2, 1))
        weights_size = len(self.weights)

        for i in range(weights_size + 1):
            if i == weights_size:
                self.zero_weight = self.zero_weight + self.learning_factor * error * self.bias
            else:
                number = 2 * self.learning_factor * error * self.patterns[offset][0][i]
                new_weights[i] = self.weights[i] + number

        return new_weights

    def activate_function_bi(self, x, bias=False):
        if bias and x > 0:
            return 1
        elif not bias and x[0] > self.threshold_bi:
            return 1
        else:
            return -1

    def train(self):
        vector_size = len(self.patterns)
        last_error = self.calculate_entire_error()
        while True:
            # caly czas zmieniac wagi !!!
            new_weights = self.update_weights(self.draw_index(vector_size))
            error = self.calculate_entire_error_for_weights(new_weights)
            self.iterations += 1
            if error < last_error:
                self.weights = new_weights
                last_error = error
                if error < self.error_threshold and self.validate():
                    break
        print("\n====> End of training <====\n")

    def draw_index(self, vector_size):
        return random.randint(0, vector_size-1)

    def propagation(self, inputs):
        # elements = inputs.copy()
        # np.insert(inputs, 1, self.bias)
        # array = np.array([[self.zero_weight]]).T
        # return self.activate_function_bi(np.dot(elements, np.append(array, self.weights)), bias=True)
        return self.activate_function_bi(np.dot(inputs, self.weights))

    def check_stop_condition(self):
        entire_error = self.calculate_entire_error()
        return not (entire_error >= self.error_threshold)

    def validate(self):
        result = 1
        print("\n====> Validation <====\n")
        for element in self.validation:
            if self.propagation(element[0]) != element[1]:
                print("NOT OK...")
                return False
        print("OK!")
        return True
