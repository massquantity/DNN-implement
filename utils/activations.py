import numpy as np
import math

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
