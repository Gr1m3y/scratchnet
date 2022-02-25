from scratchnet import network
import numpy as np


def test_neuron1():
    n = network.Neuron(np.array([0, 1]), 4)
    assert(n.weights[0] == 0)
    assert(n.weights[1] == 1)
    assert(n.bias == 4)


def test_neuron2():
    weights = np.array([0, 1])
    bias = 4
    n = network.Neuron(weights, bias)
    x = np.array([2, 3])
    assert(n.feedforward(x) == 0.9990889488055994)
