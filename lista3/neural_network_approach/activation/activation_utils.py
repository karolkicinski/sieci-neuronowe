from neural_network_approach.activation.activation_functions import relu, tanh, sigmoid, softmax


def process_activation_function_string(function_string):
    if function_string == 'relu':
        return relu
    elif function_string == 'tanh':
        return tanh
    elif function_string == 'sigmoid':
        return sigmoid
    elif function_string == 'softmax':
        return softmax
    else:
        raise Exception('Wrong activation function!')
