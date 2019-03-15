import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import numpy as np
from DNN_implementation import cifar_data


if __name__ == "__main__":
    cifar_data.init_cifar10()
    cifar_data.load_data()