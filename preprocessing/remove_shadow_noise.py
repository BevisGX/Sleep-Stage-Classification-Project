# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:49:05 2019

@author: 清华大学
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 4.0)
def judge_shadow_noise_for_single_epoch(psg_epoch):
    '''
    功能：
    识别信号中是否存在阴影噪声，若有，则处理。
    输入：
    psg_epoch:  numpy.array, shape:(30*Fs,)(1-D), 特定PSG通道的一帧信号.
    输出：
    has_shadow: bool, psg_epoch中是否含有阴影噪声
    new_epoch: numpy.array, 若无阴影噪声，则new_epoch即为psg_epoch, 若有阴影噪声，则new_epoch为去掉阴影噪声后的信号
    算法描述：
    设置噪声观察窗口（时长2s，步长1s），对窗口内信号计算直方图（bins=20）,
    统计直方图中最大值的占比，计算此占比在本帧信号内所有窗口上的均值，与阈值比较，高于阈值即为存在阴影噪声
    若存在阴影噪声，则进行噪声处理：
    对整帧信号计算直方图，对其中最大值所对应的值域范围内的信号点，取该点两侧信号点的均值
    '''
    input_shape = psg_epoch.shape
    if len(input_shape) != 1:
        raise Exception
    Fs = input_shape[0] // 30
    a = 0
    for i in range(29):
        hist = np.histogram(psg_epoch[i*Fs:i*200+2*Fs],bins=20)
        a = a + np.max(hist[0])/np.sum(hist[0])
    a = a / 29
    if a < 0.19:           # 0.19 是 经验值
        has_shadow = False
        new_epoch = psg_epoch
    else:
        has_shadow = True
        #处理
        full_hist = np.histogram(psg_epoch,bins=20)
        highest_bin_id = np.argmax(full_hist[0])
        lower_bound = full_hist[1][highest_bin_id]
        upper_bound = full_hist[1][highest_bin_id+1]
        new_epoch = np.zeros(input_shape)
        for i in range(input_shape[0]):
            if psg_epoch[i] < lower_bound or psg_epoch[i] > upper_bound:
                new_epoch[i] = psg_epoch[i]
            elif i != 0 and i != input_shape[0]-1:
                new_epoch[i] = (psg_epoch[i-1] + psg_epoch[i+1])/2
            elif i == 0:
                new_epoch[i] = psg_epoch[i+1]
            else:
                new_epoch[i] = psg_epoch[i-1]
        #plt.plot(psg_epoch)
        #plt.figure()
        #plt.plot(new_epoch)
    return has_shadow, new_epoch

'''
用例：
epoch_id = 965
eeg = np.loadtxt('H:/科技部项目/数据集/tr_A6_2017_cleaned/A6_EEG_C4A1/00000037-110799[001].csv',delimiter=',')[:,1:]

try:
    #print(judge_shadow_noise_for_single_epoch(x))
    has_shadow, new_epoch = judge_shadow_noise_for_single_epoch(eeg)
    print(has_shadow)

except Exception:
    print('expect input dimension: 1-D, but your input dimension:{}-D'.format(len(eeg.shape)))
'''

def judge_shadow_noise(psg_channel):
    '''
    功能：
    输入若干帧单通道信号，返回其中含有阴影噪声的帧号，并处理阴影噪声
    输入：
    psg_channel: numpy.array, shape: (n_samples, n_features)(2-D)
    输出：
    shadow_epochs: list, 所有含有阴影噪声的帧的编号列表
    new_channel: 对psg_channel去除阴影噪声后的结果
    '''
    input_shape = psg_channel.shape
    # print("judge shadow input", input_shape)
    new_channel = np.zeros(input_shape)
    n_samples = input_shape[0]
    shadow_epochs = []
    for i in range(n_samples):
        has_shadow, new_epoch = judge_shadow_noise_for_single_epoch(psg_channel[i])
        new_channel[i] = psg_channel[i]
        if has_shadow:
            shadow_epochs.append(i)
    return shadow_epochs, new_channel
