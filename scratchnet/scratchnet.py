import numpy as np
import network


def main():
    x = np.array([2, 3])
    nw = network.NeuralNetwork()
    print(nw.feedforward(x))


if __name__ == "__main__":
    main()
