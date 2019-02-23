import warnings
warnings.filterwarnings("ignore")
import urllib.request
import os
import pickle, sys, random
import gzip, time
import numpy as np
from sklearn.preprocessing import StandardScaler
import pdb
# sys.path.append(os.pardir)
from cifar_data import load_data
# from .cifar_data import load_data
# from .mnist_1 import load_data

'''
data, labels = [], []
for i in range(1, 6):
    with open('cifar10/data_batch_%d' % i, 'rb') as f:
        whole = pickle.load(f, encoding='bytes')
        data.extend(whole[b'data'])
        labels.extend(whole[b'labels'])

test_data, test_labels = [], []
with open('cifar10/test_batch', 'rb') as f:
    whole = pickle.load(f, encoding='bytes')
    test_data = whole[b'data']
    test_labels = np.array(whole[b'labels'])



ss = StandardScaler()
X_train = np.array(data)[:10000]
X_test = np.array(test_data)[:5000]
y_train = np.eye(10)[np.array(labels, dtype=np.int32)][:10000]
y_test = np.eye(10)[np.array(test_labels, dtype=np.int32)][:5000]

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)



ss = StandardScaler()
X_train = np.array(data)
X_test = np.array(test_data)
y_train = np.eye(10)[np.array(labels, dtype=np.int32)]
y_test = np.eye(10)[np.array(test_labels, dtype=np.int32)]

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
'''

class Network:
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
        a_hold.append(self.softmax_batch(final_layer))

        delta = self.softmax_derivative(a_hold[-1], y)
        gradient_w[-1] = np.dot(a_hold[-2].T, delta)
        gradient_b[-1] = np.sum(delta, axis=0)

        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_derivative(z_hold[-l])
            gradient_w[-l] = np.dot(a_hold[-l - 1].T, delta)
            gradient_b[-l] = np.sum(delta, axis=0)

        return gradient_w, gradient_b


class Sgd:
    def __init__(self, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size

    def update(self, weights, biases, grad_w, grad_b):
        for i, (w, gw) in enumerate(zip(weights, grad_w)):
            weights[i] = w - self.lr * gw / self.batch_size

        for i, (b, gb) in enumerate(zip(biases, grad_b)):
            biases[i] = b - self.lr * gb / self.batch_size


class Momentum:
    def __init__(self, lr, momentum, batch_size):
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.velocity is None:
            self.velocity = dict()
            self.velocity['w'] = [np.zeros(w.shape) for w in weights]
            self.velocity['b'] = [np.zeros(b.shape) for b in biases]
        '''
        self.velocity['w'] = [self.momentum * vw - self.lr * gw / self.batch_size
                              for vw, gw in zip(self.velocity['w'], grad_w)]
        self.velocity['b'] = [self.momentum * vb - self.lr * gb / self.batch_size
                              for vb, gb in zip(self.velocity['b'], grad_b)]

        for i, (w, vw) in enumerate(zip(weights, self.velocity['w'])):
            weights[i] = w + vw

        for i, (b, vb) in enumerate(zip(biases, self.velocity['b'])):
            biases[i] = b + vb
        '''

        for i, (w, vw, gw) in enumerate(zip(weights, self.velocity['w'], grad_w)):
            vw = self.momentum * vw - self.lr * gw / self.batch_size
            weights[i] = w + vw
            self.velocity['w'][i] = vw

        for i, (b, vb, gb) in enumerate(zip(biases, self.velocity['b'], grad_b)):
            vb = self.momentum * vb - self.lr * gb / self.batch_size
            biases[i] = b + vb
            self.velocity['b'][i] = vb


class NesterovMomentum:
    def __init__(self, lr, momentum, batch_size):
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.velocity is None:
            self.velocity = dict()
            self.velocity['w'] = [np.zeros(w.shape) for w in weights]
            self.velocity['b'] = [np.zeros(b.shape) for b in biases]

        for i, (w, vw, gw) in enumerate(zip(weights, self.velocity['w'], grad_w)):
            vw = self.momentum * vw - self.lr * gw / self.batch_size
            weights[i] = w + self.momentum * vw - self.lr * gw / self.batch_size
            self.velocity['w'][i] = vw

        for i, (b, vb, gb) in enumerate(zip(biases, self.velocity['b'], grad_b)):
            vb = self.momentum * vb - self.lr * gb / self.batch_size
            biases[i] = b + self.momentum * vb - self.lr * gb / self.batch_size
            self.velocity['b'][i] = vb


class Adam:
    def __init__(self, lr, batch_size, rho1=0.9, rho2=0.999):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.batch_size = batch_size
        self.iteration = 0
        self.ps = None

    def update(self, weights, biases, grad_w, grad_b):
        self.iteration += 1
        grad_w = [gw / self.batch_size for gw in grad_w]
        grad_b = [gb / self.batch_size for gb in grad_b]

        if self.ps is None:
            self.ps = {}
            self.ps['vw'] = [np.zeros(w.shape) for w in weights]
            self.ps['vb'] = [np.zeros(b.shape) for b in biases]
            self.ps['hw'] = [np.zeros(w.shape) for w in weights]
            self.ps['hb'] = [np.zeros(b.shape) for b in biases]

        self.ps['vw'] = [self.rho1 * vw + (1 - self.rho1) * gw for vw, gw in zip(self.ps['vw'], grad_w)]
        self.ps['vb'] = [self.rho1 * vb + (1 - self.rho1) * gb for vb, gb in zip(self.ps['vb'], grad_b)]
        self.ps['hw'] = [self.rho2 * hw + (1 - self.rho2) * (gw ** 2) for hw, gw in zip(self.ps['hw'], grad_w)]
        self.ps['hb'] = [self.rho2 * hb + (1 - self.rho2) * (gb ** 2) for hb, gb in zip(self.ps['hb'], grad_b)]
        unbias_vw = [vw / (1 - self.rho1 ** self.iteration) for vw in self.ps['vw']]
        unbias_vb = [vb / (1 - self.rho1 ** self.iteration) for vb in self.ps['vb']]
        unbias_hw = [hw / (1 - self.rho2 ** self.iteration) for hw in self.ps['hw']]
        unbias_hb = [hb / (1 - self.rho2 ** self.iteration) for hb in self.ps['hb']]

        for i, (w, vw, hw) in enumerate(zip(weights, unbias_vw, unbias_hw)):
            weights[i] = w - self.lr * vw / (np.sqrt(hw) + 1e-8)

        for i, (b, vb, hb) in enumerate(zip(biases, unbias_vb, unbias_hb)):
            biases[i] = b - self.lr * vb / (np.sqrt(hb) + 1e-8)



def train_DNN_minibatch(X_train, y_train, num_epochs, optimizer, batch_size, network, X_test=None, y_test=None):
    for epoch in range(num_epochs):
        start = time.time()
        random_mask = np.random.choice(len(X_train), len(X_train), replace=False)
        X_train = X_train[random_mask]
        y_train = y_train[random_mask]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]
        #    gradient_w = [np.zeros(w.shape) for w in network.weights]
        #    gradient_b = [np.zeros(b.shape) for b in network.biases]
            grad_w, grad_b = network.backprop(X_batch, y_batch)
        #    gradient_w = [gw + aw for gw, aw in zip(gradient_w, add_w)]
        #    gradient_b = [gb + ab for gb, ab in zip(gradient_b, add_b)]


            optimizer.update(network.weights, network.biases, grad_w, grad_b)  #   gradient_w, gradient_b  add_w, add_b

        #      network.weights = [weight - learning_rate * gw / batch_size for weight, gw in zip(network.weights, gradient_w)]
        #       network.biases = [bias - learning_rate * gb / batch_size for bias, gb in zip(network.biases, gradient_b)]

        if X_test is not None:
            print("Epoch {}, training_accuracy: {:>6},  validation accuracy: {:>6},  epoch time: {:.2f}s".format(
                epoch + 1,
                evaluate(X_train, y_train, network),
                evaluate(X_test, y_test, network),
                time.time() - start))
        else:
            print("Epoch {0}, training_accuracy: {1}".
                  format(epoch + 1, evaluate(X_train, y_train, network)))


def evaluate(X_val, y_val, network):
    y_pred = [np.argmax(network.predict(x)) for x in X_val]
    return np.mean([int(y_p == np.argmax(y)) for y_p, y in zip(y_pred, y_val)])



if __name__ == "__main__":
    np.random.seed(42)
 #   print("------------------", "Momentum", "----------------------")
 #   (X_train, y_train), (X_test, y_test) = load_data(normalize=False, standard=True)  # standardscale
 #   dnn = Network_mini_batch(sizes=[3072, 50, 10], activation="relu")
 #   optimizer = Momentum(lr=1e-3, momentum=0.9, batch_size=32)
#    train_DNN_minibatch(X_train, y_train, 100, optimizer, 32, dnn, X_test, y_test)
 #   print()

    print("------------------", "NesterovMomentum", "----------------------")
    (X_train, y_train), (X_test, y_test) = load_data(normalize=False, standard=True)  # standardscale
    dnn = Network_mini_batch(sizes=[3072, 50, 10], activation="relu")
    optimizer = NesterovMomentum(lr=1e-3, momentum=0.9, batch_size=32)
    train_DNN_minibatch(X_train, y_train, 100, optimizer, 32, dnn, X_test, y_test)
    print()



  #  print("------------------", "Sgd", "----------------------")
  #  (X_train, y_train), (X_test, y_test) = load_data(normalize=False, standard=True)  # standardscale
  #  dnn = Network_mini_batch(sizes=[3072, 500, 200, 10], activation="relu")
   # optimizer = Momentum(lr=0.001, momentum=0.9, batch_size=32)
   # optimizer = Adam(lr=0.001, batch_size=32)
   # optimizer = Sgd(lr=7e-4, batch_size=32)
  #  train_DNN_minibatch(X_train, y_train, 100, optimizer, 32, dnn, X_test, y_test)
  #  print()

    print("------------------", "Momentum", "----------------------")
    (X_train, y_train), (X_test, y_test) = load_data(normalize=False, standard=True)  # standardscale
    dnn = Network_mini_batch(sizes=[3072, 500, 200, 10], activation="relu")
    optimizer = Momentum(lr=7e-4, momentum=0.9, batch_size=32)
    train_DNN_minibatch(X_train, y_train, 100, optimizer, 32, dnn, X_test, y_test)
    print()

    print("------------------", "Adam", "----------------------")
    (X_train, y_train), (X_test, y_test) = load_data(normalize=False, standard=True)  # standardscale
    dnn = Network_mini_batch(sizes=[3072, 500, 200, 10], activation="relu")
    optimizer = Adam(lr=7e-4, batch_size=32)
    train_DNN_minibatch(X_train, y_train, 100, optimizer, 32, dnn, X_test, y_test)
    print()