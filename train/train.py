import random
import time
import numpy as np
from nndl.network.Network import Network, Network_mini_batch
from nndl.data.mnist_1 import load_data
from nndl.data.mnist_2 import load_data_2


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

def train_DNN_minibatch(X_train, y_train, num_epochs, learning_rate, batch_size, network, X_test=None, y_test=None):
    for epoch in range(num_epochs):
        random_mask = np.random.choice(len(X_train), len(X_train), replace=False)
        X_train = X_train[random_mask]
        y_train = y_train[random_mask]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            gradient_w = [np.zeros(w.shape) for w in network.weights]
            gradient_b = [np.zeros(b.shape) for b in network.biases]
            add_w, add_b = network.backprop(X_batch, y_batch)
            gradient_w = [gw + aw for gw, aw in zip(gradient_w, add_w)]
            gradient_b = [gb + ab for gb, ab in zip(gradient_b, add_b)]
            network.weights = [weight - learning_rate * gw / batch_size for weight, gw in zip(network.weights, gradient_w)]
            network.biases = [bias - learning_rate * gb / batch_size for bias, gb in zip(network.biases, gradient_b)]

        if X_test is not None:
            print("Epoch {0}, training_accuracy: {1},\t validation accuracy: {2}".
                  format(epoch + 1, evaluate(X_train, y_train, network), evaluate(X_test, y_test, network)))
        else:
            print("Epoch {0}, training_accuracy: {1}".
                  format(epoch + 1, evaluate(X_train, y_train, network)))

def evaluate(X_val, y_val, network):
    y_pred = [np.argmax(network.predict(x)) for x in X_val]
    return np.mean([int(y_p == np.argmax(y)) for y_p, y in zip(y_pred, y_val)])



if __name__ == "__main__":
    t0 = time.time()
    dnn = Network(sizes=[784, 30, 10], activation="relu", dropout_rate=0.5)
    (X_train, y_train),  (X_test, y_test) = load_data()
    train_DNN(X_train, y_train, 30, 0.01, dnn, X_test, y_test)
    print("total training time: ", time.time() - t0)

    '''
    t1 = time.time()
    dnn = Network_mini_batch(sizes=[784, 30, 10], activation="relu")
    (X_train, y_train), (X_test, y_test) = load_data_2(batch=True)
    train_DNN_minibatch(X_train, y_train, 30, 0.005, 32, dnn, X_test, y_test)
    print("batch total training time: ", time.time() - t1)
    '''


































