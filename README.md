# Feedforward Neural Network Implementation

## Overview:

Feedforward Neural Network implemented in pure python + [numpy](http://www.numpy.org/) . The main features are based on my [Deep Learning Notes ](https://github.com/massquantity/Deep_Learning_NOTES).



## Features: 

- [x] Xavier, He initialization
- [x] Dropout
- [x] Mini-batch / Stratified mini-batch training
- [x] Multiple optimize functions: Momentum, NesterovMomentum, RMSprop, Adam etc.
- [x] Multiple activation functions: Relu, Leaky relu, Selu, Softplus etc.
- [x] Learning rate decay
- [x] Early stopping



## How to use
```python
from DNN_implementation.data import cifar_data
from DNN_implementation.model import Network_mini_batch
from DNN_implementation.train import train_DNN_minibatch
from DNN_implementation import Momentum

(X_train, y_train), (X_test, y_test) = cifar_data.load_data(normalize=False, standard=True)  # standardscale
dnn = Network_mini_batch(sizes=[3072, 50, 10], activation="selu", alpha=0.01, dropout_rate=0.5)
optimizer = Momentum(lr=1e-3, momentum=0.9, batch_size=128)
train_DNN_minibatch(X_train, y_train, 100, optimizer, 128, dnn, X_test, y_test)
```


## Benchmarks (TODO)

| Model                                                        | mnist  | cifar-10 |
| :----------------------------------------------------------- | :----: | :------: |
| NORMAL: network size: [300, 200], activation: Sigmoid        | 0.7525 |  0.5463  |
| + Relu                                                       |        |          |
| + Relu & Xavier Initializer                                  |        |          |
| + Relu & He Initializer                                      |        |          |
| + Relu & He Initializer & Momentum                           |        |          |
| + Relu & He Initializer & Adam                               |        |          |
| + Elu & He Initializer & Adam                                |        |          |
| + Elu & He Initializer & Adam & Learning Rate Decay          |        |          |
| + Elu & He Initializer & Adam & Learning Rate Decay & Stratified mini-batch training |        |          |
| + Elu & He Initializer & Adam & Learning Rate Decay & Stratified mini-batch & Dropout |        |          |





## Learning Curve



## Lincense
MIT