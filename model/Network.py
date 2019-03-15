from .Base import NetworkBase
from ..utils.activations import *


class Network(NetworkBase):
    def __init__(self, sizes=[100, 100], activation="relu", dropout_rate=0.0):
        """
        :param sizes: list of layers
        :param activations: activation_functions
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(back_layer, forward_layer) * np.sqrt(2.0 / forward_layer)
                        for forward_layer, back_layer in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(back_layer, 1) for back_layer in sizes[1:]]
        self.dropout_rate = dropout_rate

        # TODO  activation_functions = {'sigmoid': sigmoid, 'relu': relu} tanh
        if activation.lower() == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation.lower() == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative

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
        a_hold.append(softmax(final_layer))

        # backward pass#
        delta = softmax_derivative(a_hold[-1], y)
        gradient_w[-1] = np.dot(delta, a_hold[-2].T)
        gradient_b[-1] = delta

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_derivative(z_hold[-l])
            gradient_w[-l] = np.dot(delta, a_hold[-l - 1].T)
            gradient_b[-l] = delta

        return gradient_w, gradient_b



class Network_mini_batch(NetworkBase):
    def __init__(self, sizes=[100, 100], activation="relu"):
        """
        :param sizes:
        :param activation:
        """
        super().__init__(sizes, activation)
        self.weights = [np.random.randn(forward_layer, back_layer) * np.sqrt(2.0 / forward_layer) \
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
        a_hold.append(softmax_batch(final_layer))

        delta = softmax_derivative(a_hold[-1], y)
        gradient_w[-1] = np.dot(a_hold[-2].T, delta)
        gradient_b[-1] = np.sum(delta, axis=0)

        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_derivative(z_hold[-l])
            gradient_w[-l] = np.dot(a_hold[-l - 1].T, delta)
            gradient_b[-l] = np.sum(delta, axis=0)

        return gradient_w, gradient_b

