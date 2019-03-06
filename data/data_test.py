import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(os.pardir)
import time
import numpy as np
import pdb
from cifar_data import load_data
from evaluate.evaluate import evaluate_batch
from utils.optimizers import *
from utils.activations import *
from train import data_generator
from model.Base import NetworkBase
# from .cifar_data import load_data
# from .mnist_1 import load_data


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


def train_DNN_minibatch(X_train, y_train, num_epochs, optimizer, batch_size, network,
                        X_test=None, y_test=None, batch_mode="normal"):  # balance
    for epoch in range(num_epochs):
        start = time.time()
        if batch_mode == "normal":
            random_mask = np.random.choice(len(X_train), len(X_train), replace=False)
            X_train = X_train[random_mask]
            y_train = y_train[random_mask]

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i: i + batch_size]
                y_batch = y_train[i: i + batch_size]
                grad_w, grad_b = network.backprop(X_batch, y_batch)
                optimizer.update(network.weights, network.biases, grad_w, grad_b)

        elif batch_mode == "balance":
            data = data_generator.datainit(X_train, y_train, batch_size)
            if np.ndim(y_train) > 1:
                labels = y_train.argmax(axis=1)
            else:
                labels = y_train
            classes, class_counts = np.unique(labels, return_counts=True)
            n_batches = class_counts[0] // (batch_size // len(classes)) + 1

            for _ in range(n_batches):
                X_batch, y_batch = data.generate_batch()
                grad_w, grad_b = network.backprop(X_batch, y_batch)
                optimizer.update(network.weights, network.biases, grad_w, grad_b)

        if X_test is not None:
            train_loss, train_accuracy = evaluate_batch(X_train, y_train, network)
            test_loss, test_accuracy = evaluate_batch(X_test, y_test, network)
            print("Epoch {}, training loss: {:.4f}, training accuracy: {:.4f},  \n"
                  "\t validation loss: {:.4f}, validation accuracy: {:.4f},  "
                  "epoch time: {:.2f}s ".format(
                   epoch + 1,
                   train_loss,
                   train_accuracy,
                   test_loss,
                   test_accuracy,
                   time.time() - start))
        else:
            train_loss, train_accuracy = evaluate_batch(X_train, y_train, network)
            print("Epoch {0}, training loss: {1}, training accuracy: {2}, "
                  "epoch time: {3}s".format(
                   epoch + 1,
                   train_loss,
                   train_accuracy,
                   time.time() - start))



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
    optimizer = NesterovMomentum(lr=1e-3, momentum=0.9, batch_size=128)
    train_DNN_minibatch(X_train, y_train, 100, optimizer, 128, dnn, X_test, y_test)
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