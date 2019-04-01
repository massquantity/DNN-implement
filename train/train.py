import random
import time
import numpy as np
from . import data_generator
from ..evaluate import evaluate, evaluate_batch


def train_DNN(X_train, y_train, num_epochs, learning_rate, network, X_val=None, y_val=None):
    gradient_w = [np.zeros(w.shape) for w in network.weights]
    gradient_b = [np.zeros(b.shape) for b in network.biases]
    
    for epoch in range(num_epochs):
        index = np.arange(len(X_train))
        random.shuffle(index)
     #   random_mask = np.random.choice(len(X_train), len(X_train), replace=False)
        for idx in index:
            add_w, add_b = network.backprop(X_train[idx], y_train[idx])
            network.weights = [weight - learning_rate * gw for weight, gw in zip(network.weights, add_w)]
            network.biases = [bias - learning_rate * gb for bias, gb in zip(network.biases, add_b)]

         #   gradient_w = [gw + aw for gw, aw in zip(gradient_w, add_w)]
         #   gradient_b = [gb + ab for gb, ab in zip(gradient_b, add_b)]
         #   network.weights = [weight - learning_rate * gw for weight, gw in zip(network.weights, gradient_w)]
        #    network.biases = [bias - learning_rate * gb for bias, gb in zip(network.biases, gradient_b)]

        if X_val:
            print("Epoch {0}, training_accuracy: {1},\t validation accuracy: {2}".
                  format(epoch + 1, evaluate(X_train, y_train, network), evaluate(X_val, y_val, network)))
        else:
            print("Epoch {0}, training_accuracy: {1}".
                  format(epoch + 1, evaluate(X_train, y_train, network)))


def train_DNN_minibatch(X_train, y_train, num_epochs, optimizer, batch_size, network,
                        X_test=None, y_test=None, batch_mode="normal", **kwargs):  # balance
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

        if kwargs:
            param, param_value = list(kwargs.items())[0]
            if param == "lr_decay_rate":
                optimizer.lr *= param_value
            print("learning rate: ", optimizer.lr)

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
                   time.time() - start), '\n')
        else:
            train_loss, train_accuracy = evaluate_batch(X_train, y_train, network)
            print("Epoch {0}, training loss: {1}, training accuracy: {2}, "
                  "epoch time: {3}s".format(
                   epoch + 1,
                   train_loss,
                   train_accuracy,
                   time.time() - start))



































