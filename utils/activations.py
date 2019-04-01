import numpy as np


class Sigmoid:
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z):
        return self.forward(z) * (1 - self.forward(z))


class Tanh:
    def forward(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - np.power(np.tanh(z), 2)


class ReLU:
    def forward(self, z):
        return np.maximum(z, 0)

    def derivative(self, z):
        mask = (z <= 0)
        dout = np.ones(z.shape)
        dout[mask] = 0.0
        return dout


class Softplus:
    def forward(self, z):
        return np.log(1 + np.exp(z))

    def derivative(self, z):
        return 1.0 / (1.0 + np.exp(-z))


class Softmax:
    @staticmethod
    def forward(z):
        if np.ndim(z) == 1:
            z = z - np.max(z)
            return np.exp(z) / np.sum(np.exp(z))
        elif np.ndim(z) == 2:
            z = z - np.max(z, axis=1, keepdims=True)
            return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def derivative_with_cross_entropy(self, a, b):
        return a - b






'''
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    mask = (z <= 0)
    dout = np.ones(z.shape)
    dout[mask] = 0.0
    return dout

def softmax(z):
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z))

def softmax_batch(z):
    z = z.T
    z = z - np.max(z, axis=0)
    t = np.exp(z) / np.sum(np.exp(z), axis=0)
    return t.T

def softmax_derivative(a, b):
    return a - b
'''

