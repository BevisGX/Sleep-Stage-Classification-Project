# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:43:50 2019

@author: 清华大学
"""

from scipy.signal import butter, filtfilt

def _bandpass(data,fc_low=0.5,fc_high=30,fs=200,order=2):
    '''
    功能：利用Butterworth带通滤波器去除信号中的基线漂移
    输入：
    data: np.array, shape:(30*Fs,)(1-D), 待滤波信号
    fc_low: float, 带通滤波器下限截止频率（赫兹）
    fc_high: float, 带通滤波器上限截止频率（赫兹）
    fs: data的采样频率
    order: 滤波器阶数
    输出：
    sig_filtered: 滤波结果
    '''
    b,a = butter(order, [float(fc_low)*2/fs, float(fc_high)*2/fs], 'pass')
    sig_filtered = filtfilt(b,a,data,axis=0)
    return sig_filtered