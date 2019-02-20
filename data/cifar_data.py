import os
import shutil
import gzip, tarfile
import pickle
import subprocess
import urllib.request
import numpy as np


pardir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(pardir + "/cifar_raw_data"):
	os.makedirs(pardir + "/cifar_raw_data")
dataset_dir = pardir + "/cifar_raw_data/"


def _download():
	print("Downloading...")
	if not os.path.exists("cifar-10-python.tar.gz") and \
		not os.path.exists("cifar_raw_data/cifar-10-python.tar.gz"):
		subprocess.call(
			'wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"', 
			shell=True)

	#	file_path = dataset_dir + "cifar10.tar.gz"
	#	urllib.request.urlretrieve(
	#		"https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", 
	#		file_path)
		print("Download Done ! \n")

	else:
		print("Dataset already downloaded.")


def _extract_data():
	# extract data
	file_names = ["cifar-10-batches-py/data_batch_%d" % i for i in range(1, 6)] + \
 					["cifar-10-batches-py/test_batch"]
	with tarfile.open("cifar-10-python.tar.gz") as tar:
		for file in file_names:
			tar.extract(file, path=os.path.abspath('.'))
	os.rename("cifar-10-batches-py", "cifar10")
	shutil.move("cifar-10-python.tar.gz", "cifar_raw_data/")
	print("Directory renamed...")


def init_cifar10():
	_download()
	_extract_data()
	print("Done!")


class standardscale:
	def __init__(self):
		self.mean = None
		self.std = None
        
	def fit_transform(self, X):
		self.mean = np.mean(X, axis=0)   # .astype(np.float32)
		self.std = np.std(X, axis=0)   # .astype(np.float32)
		return (X - self.mean) / self.std
        
	def transform(self, X):
		return (X - self.mean) / self.std


def _one_hot(labels):
	return np.eye(10)[np.array(labels, dtype=np.int32)]


def load_data(normalize=False, standard=True, one_hot=True):
	if not os.path.exists(dataset_dir):
		init_cifar10()

	data, labels = [], []
	for i in range(1, 6):
		with open('cifar10/data_batch_%d' % i, 'rb') as f:
			whole = pickle.load(f, encoding='bytes')
			data.extend(whole[b'data'])
			labels.extend(whole[b'labels'])

	test_data, test_labels = [], []
	with open('cifar10/test_batch', 'rb') as f:
		whole = pickle.load(f, encoding='bytes')
		test_data = whole[b'data']
		test_labels = np.array(whole[b'labels'])

	X_train = np.array(data).astype(np.float32)
	X_test = np.array(test_data).astype(np.float32)

	if normalize:
		X_train = X_train / 255.0
		X_test = X_test / 255.0

	if one_hot:
		y_train = _one_hot(labels)
		y_test = _one_hot(test_labels)

	if standard:
		ss = standardscale()
		X_train = ss.fit_transform(X_train)
		X_test = ss.transform(X_test)

	return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
	init_cifar10()
	load_data()