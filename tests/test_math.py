from scratchnet import calc
import numpy as np


def test_sigmoid1():
    assert(calc.sigmoid(0) == 0.5)


def test_sigmoid2():
    assert(calc.sigmoid(6) > 0.99)


def test_sigmoid3():
    assert(calc.sigmoid(-6) < 0.01)


def test_mse1():
    assert(calc.mse_loss(np.array([1, 0, 0, 1]),
                         np.array([0, 0, 0, 0])) == 0.5)
