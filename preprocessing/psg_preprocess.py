# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:40:22 2019

@author: 清华大学
"""

import os
import numpy as np
from preprocessing.remove_shadow_noise import judge_shadow_noise
from preprocessing.remove_baseline_drift import _bandpass
from preprocessing.remove_ecg_artifacts import get_qrs_peak, ecg_artifact_removal


def psg_preprocess(psg_channel, ecg_signal):
    '''

    :param psg_channel: PSG信号，例如C3、C4信号等等
    :param ecg_signal:
    :return:
    '''

    # 1. 去噪
    _, new_channel = judge_shadow_noise(psg_channel)
    return new_channel
    # n_samples = new_channel.shape[0]
    # sig_filtered = np.zeros(new_channel.shape)
    # new_sig = np.zeros(new_channel.shape)
    # for i in range(n_samples):
    #     # 2. 去基线漂移
    #     sig_filtered[i] = _bandpass(new_channel[i])
    #     # 3. 去伪迹
    #     qrs_index_list = get_qrs_peak(ecg_signal[i])
    #     new_sig[i] = ecg_artifact_removal(sig_filtered[i], qrs_index_list)
    # return new_sig



# if __name__ :
#     prefix = 'H:/科技部项目/数据集/tr_A6_2017_cleaned_without_ns/tr_A6_2017_Adult/'
#     folders = ['train/','validation/','test/']
#     channels = ['EEG_C3A2','EEG_C4A1','EOG_Left','EOG_Right','EMG_Chin']
#     for folder in folders:
#         subjects = []
#         files = os.listdir(prefix + folder)
#         for file in files:
#             subject_id = int(file.split('_')[0])
#             if subject_id not in subjects:
#                 subjects.append(subject_id)
#         for subject in subjects:
#             for channel in channels:
#                 new_sig = psg_preprocess(prefix+folder+str(subject)+'_'+channel+'_Alice6.csv')
#                 np.savetxt('H:/科技部项目/数据集/tr_A6_2017_cleaned_without_ns/tr_A6_2017_Adult_preprocessed/'+folder+str(subject)+'_'+channel+'_Alice6.csv',new_sig,delimiter=',')
#                 print(folder,subject,channel,'done!')