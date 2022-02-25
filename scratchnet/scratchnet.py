import numpy as np


def sigmoid(x):
    """returns the sigmoid function of x"""
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        return(sigmoid(np.dot(self.weights, inputs) + self.bias))


class NeuralNetwork():
    """
    Simple neural network class consisting of the following:
    - 2 inputs
    - 2 neurons in a hidden layer
    - 1 neuron for the output layer
    """
    def __init__(self, weights=np.array([0, 1]), bias=0):
        self.weights = weights
        self.bias = bias

        # Define the layers
        self.h1 = Neuron(self.weights, self.bias)
        self.h2 = Neuron(self.weights, self.bias)
        self.o1 = Neuron(self.weights, self.bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def main():
    x = np.array([2, 3])
    nw = NeuralNetwork()
    print(nw.feedforward(x))


if __name__ == "__main__":
    main()
