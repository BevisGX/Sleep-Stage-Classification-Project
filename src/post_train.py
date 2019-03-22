# coding=utf-8
"""
This script is used to integrate 5 models for each channel
"""

import os
import numpy as np
from keras.models import load_model
from preprocessing.npzLoader import loadNData
from preprocessing.timeDistributed import create_ngram_set
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from postprocessing.integrate import integrate_weight, train_integrate_nn


def test():
    # load data
    test_dir = "../data/A6_exp_data/processed/bad data/"
    channels = [
            'EEG C3-A2',
            'EEG C4-A1',
            'EOG LOC-A2',
            'EOG ROC-A2',
            'EMG Chin'
        ]

    test_x, test_y = loadNData(test_dir, channels)

    time_step = 3
    test_x = create_ngram_set(test_x, num_gram=time_step)

    c3_x = test_x[:, :, :, 0]
    c4_x = test_x[:, :, :, 1]
    loc_x = test_x[:, :, :, 2]
    roc_x = test_x[:, :, :, 3]
    emg_x = test_x[:, :, :, 4]

    n_classes = 6
    n_channels = 1
    n_features = test_x.shape[2]
    c3_x = np.reshape(c3_x, (c3_x.shape[0], 3, n_features, n_channels))
    c4_x = np.reshape(c4_x, (c4_x.shape[0], 3, n_features, n_channels))
    loc_x = np.reshape(loc_x, (loc_x.shape[0], 3, n_features, n_channels))
    roc_x = np.reshape(roc_x, (roc_x.shape[0], 3, n_features, n_channels))
    emg_x = np.reshape(emg_x, (emg_x.shape[0], 3, n_features, n_channels))

    # load model
    c3_model_dir = "../models/c3_channels_sleepNet"
    c3_model_list = os.listdir(c3_model_dir)
    c3_model_list.sort()
    c3_model_path = os.path.join(c3_model_dir, c3_model_list[0])
    c3_model = load_model(c3_model_path)

    c4_model_dir = "../models/c4_channels_sleepNet"
    c4_model_list = os.listdir(c4_model_dir)
    c4_model_list.sort()
    c4_model_path = os.path.join(c4_model_dir, c4_model_list[0])
    c4_model = load_model(c4_model_path)

    loc_model_dir = "../models/loc_channels_sleepNet"
    loc_model_list = os.listdir(loc_model_dir)
    loc_model_list.sort()
    loc_model_path = os.path.join(loc_model_dir, loc_model_list[0])
    loc_model = load_model(loc_model_path)

    roc_model_dir = "../models/roc_channels_sleepNet"
    roc_model_list = os.listdir(roc_model_dir)
    roc_model_list.sort()
    roc_model_path = os.path.join(roc_model_dir, roc_model_list[0])
    roc_model = load_model(roc_model_path)

    emg_model_dir = "../models/emg_channels_sleepNet"
    emg_model_list = os.listdir(emg_model_dir)
    emg_model_list.sort()
    emg_model_path = os.path.join(emg_model_dir, emg_model_list[0])
    emg_model = load_model(emg_model_path)

    all_model_dir = "../models/5_channels_1"
    all_model_list = os.listdir(all_model_dir)
    all_model_list.sort()
    all_model_path = os.path.join(all_model_dir, all_model_list[0])
    all_model = load_model(all_model_path)

    # predict
    c3_pred_y = c3_model.predict(c3_x)
    c4_pred_y = c4_model.predict(c4_x)
    loc_pred_y = loc_model.predict(loc_x)
    roc_pred_y = roc_model.predict(roc_x)
    emg_pred_y = emg_model.predict(emg_x)
    all_pred_y = all_model.predict(test_x)
    pred_y = []
    for i in range(len(c3_pred_y)):
        pred_y.append([c3_pred_y[i], c4_pred_y[i], roc_pred_y[i], loc_pred_y[i], emg_pred_y[i]])
    pred_y = np.asarray(pred_y)
    print("pred_y shape = ", pred_y.shape)

    # for emg_weight in [0, 0.2, 0.4, 0.6, 0.8]:
    #     weights = [0.792, 0.824, 0.777, 0.800, emg_weight] # 0.585
    #     integrate_pred_label = integrate_weight(pred_y, weights)
    train_integrate_nn(pred_y, test_y)



    # #evaluate
    # print("integrate model")
    # valid_cm = confusion_matrix(test_y, integrate_pred_label)
    # print(valid_cm)
    #
    # scores = accuracy_score(test_y, integrate_pred_label)
    # print('Test accuracy:', scores)
    #
    # target_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    # report = classification_report(test_y, integrate_pred_label, target_names=target_names)
    # print(report)
    #
    #
    # all_pre_label = np.argmax(all_pred_y, axis=1)
    #
    # print("all channels model:")
    # valid_cm = confusion_matrix(test_y, all_pre_label)
    # print(valid_cm)
    #
    # scores = accuracy_score(test_y, all_pre_label)
    # print('Test accuracy:', scores)
    #
    # target_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    # report = classification_report(test_y, all_pre_label, target_names=target_names)
    # print(report)

if __name__ :
    test()
