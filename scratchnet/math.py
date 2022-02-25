import numpy as np
"""useful general math functions for the network"""


def sig(x):
    """returns sigmoid function of x"""
    return 1 / (1 + np.exp(-x))


def mse_loss(y_true, y_pred):
    """returns the mean squares loss of y_true and y_pred"""
    return ((y_true - y_pred) ** 2).mean()
