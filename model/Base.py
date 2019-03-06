from abc import ABCMeta, abstractmethod
from utils.activations import *

class NetworkBase(metaclass=ABCMeta):
    def __init__(self, sizes, activation):
        self.sizes = sizes
        self.num_layers = len(sizes)
        if activation.lower() == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation.lower() == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def backprop(self):
        raise NotImplementedError