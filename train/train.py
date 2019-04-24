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
                        X_test=None, y_test=None, batch_mode="normal", **kwargs):
    """
    :param num_epochs: epoch numbers
    :param optimizer:  optimize function
    :param batch_size: batch size
    :param network:    network model
    :param batch_mode: batch generation mode, batch_mode means generate batch based on classes
    :param **kwargs:   miscellaneous arguments
    :return:
    """
    allowed_kwargs = {'early_stopping',
                      'tolerance',
                      'patience',
                      'metrics',
                      'restore_best_params',
                      'lr_decay_mode',
                      'lr_decay_rate',
                      'evaluate'}
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError('Keyword argument not understood:', kwarg)


    if kwargs.get("early_stopping"):
        print("use early stopping")
        early_stopping = True
        tolerance = kwargs.get("tolerance", 0.0)
        patience = kwargs.get("patience", 5)
        metrics = kwargs.get("metrics", "loss")
        best_metrics = np.infty
        restore_best_params = kwargs.get("restore_best_params", False)
        count = 0
    else:
        early_stopping = False

    if kwargs.get("lr_decay_mode") == "normal" or "exponential":
        lr_decay = kwargs.get("lr_decay_mode")
        lr_decay_rate = kwargs.get("lr_decay_rate", 0.99)
    else:
        lr_decay = False

    for epoch in range(1, num_epochs + 1):
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

        if lr_decay == "normal" and epoch > 10 and optimizer.lr > 5e-5:
            optimizer.lr *= lr_decay_rate
            print("learning rate: ", optimizer.lr)
        elif lr_decay == "exponential" and optimizer.lr > 5e-5:
            optimizer.lr *= lr_decay_rate ** (epoch / 20)
            print("learning rate: ", optimizer.lr)

        if early_stopping:
            metrics_epoch = evaluate_batch(X_test, y_test, network)
            if metrics == "loss":
                current_metrics = metrics_epoch[0]
            elif metrics == "accuracy":
                current_metrics = - metrics_epoch[1]
            if current_metrics < best_metrics - tolerance:
                best_metrics = current_metrics
                loss, accuracy = metrics_epoch[0], metrics_epoch[1]
                count = 0
                model_params = network.params
            else:
                count += 1

            if count > patience:

                print("Early Stopping in epoch %d , best loss is %.4f, best accuracy is %.4f" % (
                    epoch, loss, accuracy))
                break

        if kwargs.get('evaluate'):
            if X_test is not None:
                train_loss, train_accuracy = evaluate_batch(X_train, y_train, network)
                test_loss, test_accuracy = evaluate_batch(X_test, y_test, network)
                print("Epoch {}, training loss: {:.4f}, training accuracy: {:.4f},  \n"
                      "\t validation loss: {:.4f}, validation accuracy: {:.4f},  "
                      "epoch time: {:.2f}s ".format(
                       epoch,
                       train_loss,
                       train_accuracy,
                       test_loss,
                       test_accuracy,
                       time.time() - start), '\n')
            else:
                train_loss, train_accuracy = evaluate_batch(X_train, y_train, network)
                print("Epoch {0}, training loss: {1}, training accuracy: {2}, "
                      "epoch time: {3}s".format(
                       epoch,
                       train_loss,
                       train_accuracy,
                       time.time() - start))

    if restore_best_params:
        network.weights, network.biases = model_params




