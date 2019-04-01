from DNN_implementation.data import cifar_data
from DNN_implementation.model import Network, Network_mini_batch
from DNN_implementation.train import train_DNN_minibatch
from DNN_implementation import Sgd, Momentum, NesterovMomentum, Adam
from DNN_implementation import Sigmoid, ReLU


if __name__ == "__main__":
    print("------------------", "Momentum", "----------------------")
    (X_train, y_train), (X_test, y_test) = cifar_data.load_data(normalize=False, standard=True)  # standardscale
    dnn = Network_mini_batch(sizes=[3072, 50, 10], activation="selu", alpha=0.01)
    optimizer = Momentum(lr=1e-3, momentum=0.9, batch_size=128)
    train_DNN_minibatch(X_train, y_train, 100, optimizer, 128, dnn, X_test, y_test,
                        early_stopping=True, tolerance=0.0, metrics="accuracy")  # lr_decay_rate=0.95
    print()






