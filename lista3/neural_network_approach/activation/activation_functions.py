import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x)) / np.square((np.exp(-x) + 1))
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.square(np.tanh(x))
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return np.maximum(x, 0).astype(int)
    return np.maximum(x, 0)


def softmax(x, derivative=False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)
