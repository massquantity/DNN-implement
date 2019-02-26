import numpy as np

class datainit:
    def __init__(self, X, y, batch_size):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.__init_batch()

    def __init_batch(self):
        if len(self.y.shape) > 1:
            labels = self.y.argmax(axis=1)
        else:
            labels = self.y
        classes, y_indices, class_counts = np.unique(labels, return_inverse=True,
                                                     return_counts=True)
        self.n_classes = classes.shape[0]
        self.class_indices = np.split(np.argsort(y_indices, kind="mergesort"),
                                      np.cumsum(class_counts)[:-1])

        self.pre_batch = class_counts[0] // (self.batch_size // self.n_classes)
        #    self.n_batches = math.ceil(len(y) / batch_size)
        self.num_per_class = self.batch_size // self.n_classes
        self.intra_batch_indices = dict(zip(list(range(self.n_classes)), [0] * self.n_classes))
        self.perm_indices_class = dict()
        for i in range(self.n_classes):
            self.perm_indices_class[i] = self.class_indices[i].take(
                np.random.permutation(class_counts[i]), mode="clip")

    def generate_batch(self):
        batch = []
        if np.all(np.array(list(self.intra_batch_indices.values())) <= self.pre_batch):
            for i in range(self.n_classes):
                j = self.intra_batch_indices[i]
                batch.extend(self.perm_indices_class[i][j * self.num_per_class:
                                                        (j + 1) * self.num_per_class])
                self.intra_batch_indices[i] += 1
            remain_batch = np.random.choice(len(self.y), self.batch_size - len(batch), replace=False)
            batch.extend(remain_batch)

        else:
            raise IndexError("indices out of bounds...")

        batches = np.random.permutation(batch).astype(np.int32)
        return self.X[batches], self.y[batches]