# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:34:06 2019

@author: 清华大学
"""

import numpy as np
from preprocessing import My_Detect_QRS
from preprocessing.Correct_Peak_Index import correct_peaks, _remove_close_index, _bandpass
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (50.0, 4.0)

def get_qrs_peak(signal_ECGII):
    fre_ECGII = signal_ECGII.shape[0] // 30
    # 利用XQRS类执行初步检测
    qrs_index_array_ori = My_Detect_QRS.xqrs_detect(sig = signal_ECGII, fs = fre_ECGII)
    # 保证qrs波索引数组有序
    qrs_index_array_ori = np.sort(qrs_index_array_ori)
    # 对qrs波索引数组进行第一步修正，调整至局部极值
    max_bpm = 200 # 定义最快心跳200 BPM
    thre = float(60.0/max_bpm) # 定义了最快心跳，便得到了最短的RR间期 
    search_radius = int(fre_ECGII * 60 / max_bpm)
    # 1-25Hz可以充分保留心电的原始形态
    ECG_filtered = _bandpass(signal_ECGII, fc_low = 1, fc_high = 25, fre = fre_ECGII, order = 2)
    qrs_index_array_correct_1 = correct_peaks(sig=ECG_filtered, peak_inds = qrs_index_array_ori,
                                              search_radius=search_radius, smooth_window_size=
                                              int(5.0/12.0 * fre_ECGII))
    # 对qrs波索引数组进行第二步修正，删除距离太近的索引
    qrs_index_array = _remove_close_index(qrs_index_array_correct_1, 
                                          fs = fre_ECGII, Thre = thre)
    # qrs索引数组中的首元素可能为负，则修正一下
    if qrs_index_array[0] < 0:
        # qrs_index_array[0] = 0
        qrs_index_list = list(qrs_index_array)
        del qrs_index_list[0]
        qrs_index_array = np.array(qrs_index_list)
    qrs_index_list = list(qrs_index_array)
    return qrs_index_list

def ecg_artifact_removal(eeg,qrs_index_list):
    Fs = eeg.shape[0] // 30
    new_eeg = eeg.copy()
    for i in qrs_index_list:
        start = int(max(i - 0.05 * Fs, 0))
        end = int(min(i + 0.05 * Fs, eeg.shape[0]-1))
        if min(2*end-start,eeg.shape[0]) - max(end,0) != end - start:
            new_eeg[start:end] = new_eeg[max(2*start-end,0):min(start,eeg.shape[0])]
        elif min(start,eeg.shape[0]) - max(2*start-end,0) != end - start:
            new_eeg[start:end] = new_eeg[max(end,0):min(2*end-start,eeg.shape[0])]
        else:
            new_eeg[start:end] = (new_eeg[max(end,0):min(2*end-start,eeg.shape[0])]+new_eeg[max(2*start-end,0):min(start,eeg.shape[0])])/2
    return new_eeg

'''
用例：
epoch_id = 699
ecg = np.loadtxt('H:/科技部项目/数据集/tr_A6_2017_cleaned/A6_ECG_II/00000115-110799[001].csv',delimiter=',')[epoch_id,1:]
eeg = np.loadtxt('H:/科技部项目/数据集/tr_A6_2017_cleaned/A6_EEG_C3A2/00000115-110799[001].csv',delimiter=',')[epoch_id,1:]

qrs_index_list = get_qrs_peak(ecg)
new_eeg = ecg_artifact_removal(eeg,qrs_index_list)
plt.figure()
plt.subplot(311)
plt.plot(eeg)
plt.subplot(312)
plt.plot(new_eeg)
plt.subplot(313)
plt.plot(ecg)
'''