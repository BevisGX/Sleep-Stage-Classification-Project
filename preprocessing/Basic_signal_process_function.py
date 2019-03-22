# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:36:53 2018

@author: Qingyan Zou
"""
"""
说明：本文件中的函数是处理ECG信号过程中用到的两个子函数：
(1) get_filter_gain:获取滤波增益
(2) find_local_peaks:找到局部波峰点
"""
import numpy as np
from scipy import signal

def get_filter_gain(b, a, f_gain, fs):
    """
    功能：给定滤波器系数，返回在特定频率上的增益
  
    参数：
    ----------
    b : 类型 list.滤波器系数列表b
    a : 类型 list.滤波器系数列表a
    f_gain : 类型 int 或 float.在频率上计算得到的增益
    fs : 类型 int 或 float.信号的采样频率
        
    """

    w, h = signal.freqz(b, a)
    w_gain = f_gain * 2 * np.pi / fs

    ind = np.where(w >= w_gain)[0][0]
    gain = abs(h[ind])

    return gain


def find_local_peaks(sig, radius):
    """
    功能：找到信号中的局部波峰点。如果一个采样点的幅值在radius范围内采样点的幅值最大，
    那么该采样点就是局部波峰点。
    如果有多个采样点具有相同的最大值，那么中间的那个采样点是局部波峰点

    输入参数：
    ----------
    sig : 类型 numpy array.输入信号的一维数组
    radius : 类型 int.定义局部波峰点的搜索半径
    
    """

    if np.min(sig) == np.max(sig):
        return np.empty(0)

    peak_inds = []

    i = 0
    while i < radius + 1:
        if sig[i] == max(sig[:i + radius]):
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    while i < len(sig):
        if sig[i] == max(sig[i - radius:i + radius]):
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    while i < len(sig):
        if sig[i] == max(sig[i - radius:]):
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    return (np.array(peak_inds))
