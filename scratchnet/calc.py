import numpy as np
"""useful general math functions for the network"""


def sigmoid(x):
    """returns sigmoid function of x"""
    return 1 / (1 + np.exp(-x))


def mse_loss(y_true, y_predicted):
    """returns the mean squares loss of y_true and y_predicted"""
    return ((y_true - y_predicted) ** 2).mean()
