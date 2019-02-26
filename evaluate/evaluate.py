import numpy as np

def evaluate(X_val, y_val, network):
    y_pred = [np.argmax(network.predict(x)) for x in X_val]
    return np.mean([int(y_p == np.argmax(y)) for y_p, y in zip(y_pred, y_val)])

def evaluate_batch(X, y, network):
    y_pred = network.softmax_batch(network.predict(X))
  #  y_prob = y_pred[np.arange(len(y)), y.argmax(axis=1)]
    y_prob = np.take_along_axis(y_pred, np.expand_dims(y.argmax(axis=1), axis=1), axis=1)
    loss = - np.sum(np.log(y_prob)) / len(y)

    y_true = np.argmax(y, axis=1)
    y_pred_index = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_true == y_pred_index)
    return loss, accuracy