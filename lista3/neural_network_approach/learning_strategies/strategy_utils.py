from neural_network_approach.learning_strategies.adadelta import Adadelta
from neural_network_approach.learning_strategies.adagrad import Adagrad
from neural_network_approach.learning_strategies.adam import Adam
from neural_network_approach.learning_strategies.momentum import Momentum
from neural_network_approach.learning_strategies.momentum_nesterov import MomentumNesterov


def process_learning_strategy_string(name):
    if name == 'momentum':
        return Momentum
    elif name == 'momentum_nesterov':
        return MomentumNesterov
    elif name == 'adagrad':
        return Adagrad
    elif name == 'adadelta':
        return Adadelta
    elif name == 'adam':
        return Adam
    else:
        raise Exception('Wrong activation function!')