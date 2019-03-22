# coding=utf-8

import numpy as np

def create_ngram_set(data, num_gram=3):
    """
    Concatenate 3 gram to 1 sample, for example, input shape = (N, 1000), output shape will be (N, 3, 1000)
    :param data:
    :param num_gram:
    :return:
    """
    num_gram -= 1
    n_feature = data.shape[1]
    n_channels = data.shape[2]
    new_data = []

    for ii in range(len(data)):
        features = []
        current = data[ii]

        for jj in range(num_gram):
            previous = np.zeros((n_feature, n_channels), dtype='float32')
            if ii - (num_gram - jj) >= 0:
                previous = data[ii - (num_gram - jj)]
            features.append(previous)

        features.append(current)
        new_data.append(np.asarray(features))
    return np.asarray(new_data)