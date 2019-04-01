import pickle
import gzip
import numpy as np

def load_data():
    with gzip.open("../data/mnist.pkl.gz", 'rb') as f:
        train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    train_inputs = [np.reshape(x, (28 * 28, 1)) for x in train_data[0]] + \
                   [np.reshape(x, (28 * 28, 1)) for x in val_data[0]]
    train_labels = [vectorized_result(y) for y in train_data[1]] + \
                   [vectorized_result(y) for y in val_data[1]]
   # valid_inputs = [np.reshape(x, (28*28, 1)) for x in val_data[0]]
   # valid_labels = [vectorized_result(y) for y in val_data[1]]
    test_inputs = [np.reshape(x, (28*28, 1)) for x in test_data[0]]
    test_labels = [vectorized_result(y) for y in test_data[1]]
    return (train_inputs, train_labels), (test_inputs, test_labels)  # (valid_inputs, valid_labels),

def vectorized_result(i):
    label = np.zeros((10, 1))
    label[i] = 1.0
    return label


