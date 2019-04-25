import numpy as np


def truncated_normal2(mean, scale, shape):
    total_num = np.multiply(*shape)
    data = []
    while len(data) != total_num:
        sample = np.random.normal(mean, scale, 1)
        while sample > mean + 2 * scale or sample < mean - 2 * scale:
            sample = np.random.normal(mean, scale, 1)
        data.append(sample)
    array = np.array(data).reshape(*shape)
    return array


def truncated_normal(mean, scale, shape):
    total_num = np.multiply(*shape)
    array = np.random.normal(mean, scale, total_num)
    for i, sample in enumerate(array):
        while sample > mean + 2 * scale or sample < mean - 2 * scale:
            sample = np.random.normal(mean, scale, 1)
            array[i] = sample
    return array.reshape(*shape)