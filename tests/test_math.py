from scratchnet import math
import numpy as np


def test_sigmoid1():
    assert(math.sig(0) == 0.5)


def test_sigmoid2():
    assert(math.sig(6) > 0.99)


def test_sigmoid3():
    assert(math.sig(-6) < 0.01)


def test_mse1():
    assert(math.mse_loss(np.array([1, 0, 0, 1]),
                         np.array([0, 0, 0, 0])) == 0.5)
