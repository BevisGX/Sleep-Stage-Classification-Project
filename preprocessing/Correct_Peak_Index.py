# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 22:41:29 2018

@author: Qingyan Zou
"""
"""
说明：本代码的功能是对XQRS类的检测结果进行修改，把索引点修正到
      局部最大值或最小值（考虑电极按反了的情况）。
"""
import numpy as np
from scipy import signal

def correct_peaks(sig, peak_inds, search_radius, smooth_window_size,
                  peak_dir='compare'):
    """
    功能：调整一组检测到的QRS波索引至局部最大值处

    输入参数：
    ----------
    sig : 类型 numpy array.信号的一维数组
    peak_inds : 类型 numpy array.原始qrs波索引的数组
    max_gap : 类型 int.原始qrs波索引可以移动的最大半径
    smooth_window_size : 类型 int.应用于信号的滑动平均滤波器的窗口大小,峰值距离是
        根据原始和平滑信号之间的差异计算的
    peak_dir : 类型 str, 可选.
        峰值移动方向有4类: 'up' or 'down', 'both', 或'compare'.
        - 'up', 峰值索引会移动到局部最大值处
        - 'down', 峰值索引会移动到局部最小值处
        - 'both', 峰值将被转移到整流信号的局部最大值
        - 'compare', 函数会尝试'up'和'down'两个选择，最终会根据平滑后的信号两个方
            向上的索引点振幅的均值选择振幅均值较大的方向

    输出参数：
    -------
    corrected_peak_inds : 类型 numpy array.修正后的qrs波索引的数组
    """
    
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)

    # 从原始信号中减去平滑后的信号
    sig = sig - smooth(sig=sig, window_size=smooth_window_size)


    # 把波峰索引调整至局部最大值处
    if peak_dir == 'up':
        shifted_peak_inds = shift_peaks(sig=sig,
                                        peak_inds=peak_inds,
                                        search_radius=search_radius,
                                        peak_up=True)
    # 把波峰索引调整至局部最小值处
    elif peak_dir == 'down':
        shifted_peak_inds = shift_peaks(sig=sig,
                                        peak_inds=peak_inds,
                                        search_radius=search_radius,
                                        peak_up=False)
    # 把波峰索引调整至整流信号局部最大值处
    elif peak_dir == 'both':
        shifted_peak_inds = shift_peaks(sig=np.abs(sig),
                                        peak_inds=peak_inds,
                                        search_radius=search_radius,
                                        peak_up=True)
    # 同时尝试'up'和'down'两个选择
    else:
        shifted_peak_inds_up = shift_peaks(sig=sig,
                                           peak_inds=peak_inds,
                                           search_radius=search_radius,
                                           peak_up=True)
        shifted_peak_inds_down = shift_peaks(sig=sig,
                                             peak_inds=peak_inds,
                                             search_radius=search_radius,
                                             peak_up=False)

        # 选择波峰点振幅均值相对较大的方向
        up_dist = np.mean(np.abs(sig[shifted_peak_inds_up]))
        down_dist = np.mean(np.abs(sig[shifted_peak_inds_down]))

        if up_dist >= down_dist:
            shifted_peak_inds = shifted_peak_inds_up
        else:
            shifted_peak_inds = shifted_peak_inds_down

    return shifted_peak_inds


def shift_peaks(sig, peak_inds, search_radius, peak_up):
    """
    功能：correct_peaks()函数的辅助函数.在一定范围内，把索引调整至局部最大或最小值
        处并返回
    
    输入参数：(与correct_peaks()函数相同的输入参数不再赘述)
    peak_up : 类型 bool.是否认为qrs波中r波波峰向上
    """
    
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)
    # 初始化调整后的索引数组
    shift_inds = np.zeros(n_peaks, dtype='int')

    # 遍历每一个索引
    for i in range(n_peaks):
        ind = peak_inds[i]
        local_sig = sig[max(0, ind - search_radius):min(ind + search_radius, sig_len-1)]

        if peak_up:
            shift_inds[i] = np.argmax(local_sig)
        else:
            shift_inds[i] = np.argmin(local_sig)

    # 可能需要调整之前的qrs波索引值
    for i in range(n_peaks):
        ind = peak_inds[i]
        if ind >= search_radius:
            break
        shift_inds[i] -= search_radius - ind

    shifted_peak_inds = peak_inds + shift_inds - search_radius

    return shifted_peak_inds


def smooth(sig, window_size):
    """
    功能：对信号应用滑动均值滤波

    输入参数：
    ----------
    sig : 类型 numpy array.需要被平滑的信号
    window_size : 类型 int.滑动均值滤波的窗口大小
    """
    
    box = np.ones(window_size)/window_size
    return np.convolve(sig, box, mode='same')

def _bandpass(data, fc_low = 1, fc_high = 25, fre = 200, order = 2): 
    # default: fc_low = 5, fc_high = 20, order = 2
    # 1-25Hz能充分保留信号的特点
    """
    功能：对信号利用带通滤波器进行处理，行保存滤波后的信号
    
    输入参数：
    ----------
    data : 类型 numpy array.需要被过滤的信号
    """
    
    b, a = signal.butter(order, [float(fc_low) * 2 / fre,
                             float(fc_high) * 2 / fre], 'pass')
    sig_filtered = signal.filtfilt(b, a, data, axis=0)
    return sig_filtered

def _remove_close_index(peak_index, fs = 200, Thre = 0.3):
    """"
    功能：删除相距太近的qrs索引点
    
    输入参数：
    ----------
    peak_index : 类型 numpy array.存储qrs索引点的numpy array
    fs : 类型 int.原始心电信号的采样频率
    Thre : 类型 float.判断阈值，单位：秒
    
    输出参数：
    ----------
    corrected_peak_index : 类型 numpy array.修改后存储qrs索引点的numpy array
    """
    
    peak_index_size = len(peak_index) 
    if peak_index_size < 2:
        return peak_index
    
    corrected_peak_index = []
    corrected_peak_index.append(peak_index[0])
    for i in range(1, peak_index_size):
        if peak_index[i] - corrected_peak_index[-1] >= int(Thre * fs):
            corrected_peak_index.append(peak_index[i])
        else:
            continue
    corrected_peak_index = np.array(corrected_peak_index)
    return corrected_peak_index
