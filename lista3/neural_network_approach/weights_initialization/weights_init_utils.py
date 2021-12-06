from neural_network_approach.weights_initialization.he import He
from neural_network_approach.weights_initialization.weights_initializer import WeightsInitializer
from neural_network_approach.weights_initialization.xavier import Xavier


def process_weights_inits_string(name):
    if name == 'xavier':
        return Xavier
    elif name == 'he':
        return He
    elif name == 'standard':
        return WeightsInitializer
    else:
        raise Exception('Wrong weights initialization method!')
