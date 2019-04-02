import numpy as np


class Sigmoid:
    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z):
        return self.__call__(z) * (1 - self.__call__(z))


class Tanh:
    def __call__(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - np.power(np.tanh(z), 2)


class ReLU:
    def __call__(self, z):
        return np.maximum(z, 0)

    def derivative(self, z):
        mask = (z <= 0)
        dout = np.ones(z.shape)
        dout[mask] = 0.0
        return dout


class LeakyReLU:
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def __call__(self, z):
        return np.where(z >= 0, z, self.alpha * z)

    def derivative(self, z):
        return np.where(z >= 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, z):
        return np.where(z >= 0, z, self.alpha * (np.exp(z) - 1))

    def derivative(self, z):
        return np.where(z >= 0, 1, self.alpha * np.exp(z))


class Selu:
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, z):
        return self.scale * np.where(z >= 0, z, self.alpha * (np.exp(z) - 1))

    def derivative(self, z):
        return self.scale * np.where(z >= 0, 1, self.alpha * np.exp(z))


class Softplus:
    def forward(self, z):
        return np.log(1 + np.exp(z))

    def __call__(self, z):
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



