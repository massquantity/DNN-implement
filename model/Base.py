from abc import ABCMeta, abstractmethod
from ..utils.activations import *

class NetworkBase(metaclass=ABCMeta):
    def __init__(self, sizes, activation, last_layer, **kwargs):
        self.sizes = sizes
        self.num_layers = len(sizes)
        if activation.lower() == "sigmoid":
            self.activation = Sigmoid()
        #    self.activation_derivative = sigmoid_derivative
        elif activation.lower() == "relu":
            self.activation = ReLU()
        #    self.activation_derivative = relu_derivative
        elif activation.lower() == "tanh":
            self.activation = Tanh()
        elif activation.lower() == "softplus":
            self.activation = Softplus()
        elif activation.lower() == "leaky_relu" or "leakyrelu":
            if "alpha" in kwargs:
                self.activation = LeakyReLU(kwargs.get("alpha"))
            else:
                self.activation = LeakyReLU()
        elif activation.lower() == "elu":
            if "alpha" in kwargs:
                self.activation = ELU(kwargs.get("alpha"))
            else:
                self.activation = ELU()
        elif activation.lower() == "selu":
            self.activation = Selu()

        if last_layer.lower() == "softmax":
            self.last_layer = Softmax()

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def backprop(self):
        raise NotImplementedError