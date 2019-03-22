# coding=utf-8
"""
This tool is used to extract .npz files
"""

import numpy as np
import os

def loadData(feature_file, label_file, channels):
    """
    Load npz data from npz file, return shape as (num_samples, num_features, num_channels)
    :param feature_file, label_file: string of file's path
    :param channels: required channel names (a list of strings)
    :return:
        features: numpy array in shape (num_samples, num_features, num_channels)
        labels: numpy array in shape (nSamples)
    """
    ff = np.load(feature_file)
    lf = np.load(label_file)
    features = []
    labels = lf['stage_labels']
    sfreq = ff['sampling rate'][0]
    for ch in channels:
        if ch not in ff['channels']:
            raise Exception("Request channel is not in the channels lists")
        features.append(ff['features'][ff['channels'].tolist().index(ch)])
    features = np.asarray(features)
    k, M, N = features.shape
    if M != len(labels):
        raise Exception('The number of features is not equal to the number of labels')
    if k != len(channels):
        raise Exception('The number of channels does not match the requirements')
    # if N != 30 * sfreq:
    #     raise Exception('The number of points does not match the sampling frequency')
    ftmp = []
    for i in range(M):
        ftmp.append(features[:, i, :].transpose())
    features = np.asarray(ftmp)
    ftmp = []
    ltmp = []
    for i in range(M):
        if labels[i] != -1:
            ltmp.append(labels[i])
            ftmp.append(features[i, :])
    labels = np.asarray(ltmp)
    features = np.asarray(ftmp)
    print("Successfully extracted file %s \nand %s" % (feature_file, label_file))
    print(features.shape)
    return features, labels

def loadNData(data_dir, channels, include=-1, exclude=-1):
    """
    load several npz data file from data directory, from n=start to n=end
    :param data_dir: string of directory's name
    :param channels: required channel names (a list of strings)
    :param include: default -1, if not, the function will only load data which index equals include
    :param exclude: default -1, if not, the function will load data except which index equals exclude
    :return:
            features: numpy array in shape(num_samples, num_features, num_channels)
            labels: numpy array in shape(num_samples)
    """
    fileNum = 0
    features = []
    labels = []
    fnames = os.listdir(data_dir)
    fnames.sort()
    for i in range(len(fnames)):
        fname = fnames[i]
        if fname.split('_')[0] != 'Features':
            continue
        id = fname.split('_')[1]
        if include != -1 and id != include:
            continue
        if id == exclude:
            continue
        ff = os.path.join(data_dir, fname)      # feature file
        lf = os.path.join(data_dir, 'Labels_' + fname.split('_')[1])        #label file
        feature, label = loadData(ff, lf, channels)
        fileNum += 1
        features.append(feature)
        labels.append(label)
    features = np.vstack(features)
    labels = np.hstack(labels)
    print("loaded {} files".format(fileNum))
    return features, labels
