import random
import numpy as np
from nndl.data.mnist_1 import load_data


class Network:
    def __init__(self, sizes=[100, 100], activation="relu", dropout_rate=0.0):
        """
        :param sizes: list of layers
        :param activations: activation_functions
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(back_layer, forward_layer) * np.sqrt(2.0 / forward_layer) \
                        for forward_layer, back_layer in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(back_layer, 1) for back_layer in sizes[1:]]
        self.dropout_rate = dropout_rate

        # TODO  activation_functions = {'sigmoid': sigmoid, 'relu': relu} tanh
        if activation.lower() == "sigmoid":
            self.activation = Network.sigmoid
            self.activation_derivative = Network.sigmoid_derivative
        elif activation.lower() == "relu":
            self.activation = Network.relu
            self.activation_derivative = Network.relu_derivative

    def predict(self, a):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.activation(np.dot(w, a) + b)
            a *= (1.0 - self.dropout_rate)  ######### test dropout
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        return a

    def backprop(self, x, y):
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        # forward pass #
        a = x
        a_hold = [x]
        z_hold = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b

            self.mask = np.random.rand(*z.shape) > self.dropout_rate
            z *= self.mask
        #    z /= (1 - self.dropout_rate)

            a = self.activation(z)
            z_hold.append(z)
            a_hold.append(a)
        final_layer = np.dot(self.weights[-1], a) + self.biases[-1]
        z_hold.append(final_layer)
        a_hold.append(Network.softmax(final_layer))

        # backward pass#
        delta = Network.softmax_derivative(a_hold[-1], y)
        gradient_w[-1] = np.dot(delta, a_hold[-2].T)
        gradient_b[-1] = delta

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_derivative(z_hold[-l])
            gradient_w[-l] = np.dot(delta, a_hold[-l - 1].T)
            gradient_b[-l] = delta

        return gradient_w, gradient_b


    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def relu_derivative(z):
        mask = (z <= 0)
        dout = np.ones(z.shape)
        dout[mask] = 0.0
        return dout

    @staticmethod
    def softmax(z):
        z = z - np.max(z)
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def softmax_batch(z):
        z = z.T
        z = z - np.max(z, axis=0)
        t = np.exp(z) / np.sum(np.exp(z), axis=0)
        return t.T

    @staticmethod
    def softmax_derivative(a, b):
        return a - b


class Network_mini_batch(Network):
    def __init__(self, sizes=[100, 100], activation="relu"):
        super().__init__(sizes, activation)
        self.weights = [np.random.randn(forward_layer, back_layer) \
                        for forward_layer, back_layer in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(layer) for layer in sizes[1:]]

    def predict(self, a):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.activation(np.dot(a, w) + b)
        a = np.dot(a, self.weights[-1]) + self.biases[-1]
        return a

    def backprop(self, x, y):
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        a = x
        a_hold = [x]
        z_hold = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(a, w) + b  # batch  z = a * w + b
            a = self.activation(z)
            z_hold.append(z)
            a_hold.append(a)
        final_layer = np.dot(a, self.weights[-1]) + self.biases[-1]
        z_hold.append(final_layer)
        a_hold.append(self.softmax_batch(final_layer))
        
        delta = self.softmax_derivative(a_hold[-1], y)
        gradient_w[-1] = np.dot(a_hold[-2].T, delta)
        gradient_b[-1] = np.sum(delta, axis=0)

        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_derivative(z_hold[-l])
            gradient_w[-l] = np.dot(a_hold[-l - 1].T, delta)
            gradient_b[-l] = np.sum(delta, axis=0)

        return gradient_w, gradient_b

