import urllib.request
import os
import pickle
import gzip
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_files = ['train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz']

par_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(par_dir + "/mnist_download"):
    os.mkdir(par_dir + "/mnist_download")
dataset_dir = par_dir + "/mnist_download/"

save_file = os.path.join(dataset_dir, "mnist.pkl")

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(files):
    for file in files:
        file_path = dataset_dir + file
        if os.path.exists(file_path):
            return

        print("Downloading " + file + "...")
        urllib.request.urlretrieve(url_base + file, file_path)
        print(file + " Done.")


def _data_processing(files):
    dataset = []
    for file in files:
        if 'images' in file:
            with gzip.open(dataset_dir + file, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            #    data = data.reshape(-1, img_size)
            #    data = [x.reshape(img_size, 1) for x in data]
            #    print(data[0].shape)
                data = [np.reshape(data[i : i+784], (784, 1)) for i in range(0, len(data), 784)]
        else:
            with gzip.open(dataset_dir + file, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        dataset.append(data)
    return dataset


def init_mnist():
    _download(key_files)
    dataset = _data_processing(key_files)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _one_hot(Y):
   # T = np.zeros((Y.size, 10))
  #  for idx, row in enumerate(T):
  #      T[Y[idx]] = 1.0

    ys = []
    for i in range(len(Y)):
        single_label = np.zeros((10, 1))
        single_label[Y[i]] = 1.0
        ys.append(single_label)
    return ys


def load_data_2(normalize=True, flatten=True, one_hot=True, batch=True):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for t in [0, 2]:
    #        dataset[t] = dataset[t].astype(np.float32)
     #       dataset[t] /= 255.0
            dataset[t] = [x.astype(np.float32) / 255.0 for x in dataset[t]]

    if one_hot:
        for y in [1, 3]:
            dataset[y] = _one_hot(dataset[y])

    if batch:
        for i in range(4):
            length = len(dataset[i])
            dataset[i] = np.array(dataset[i]).reshape(length, -1)
      #      if i == 0:
       #         print(dataset[i][0].shape)

    if not flatten:
        for t in [0, 2]:
            dataset[t] = dataset[t].reshape(-1, 1, 28, 28)

    return (dataset[0], dataset[1]), (dataset[2], dataset[3])


if __name__ == "__main__":
    init_mnist()
    load_data_2()





















