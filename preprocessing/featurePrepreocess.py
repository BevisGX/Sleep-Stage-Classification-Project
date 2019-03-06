"""
EDF data feature engineering before training, by downsample filter and  fast Fourier transform,
or short time Fourier transform
"""

from scipy.signal import decimate, stft
from numpy.fft import rfft
from preprocessing.npzLoader import loadData, loadNData
import os
import numpy as np
import matplotlib.pyplot as plt


feature_file = "../data/A6_exp_data/Features_00000020.npz"
label_file = "../data/A6_exp_data/Labels_00000020.npz"
data_dir = "../data/A6_exp_data"
channels = [
    # 'EEG F3-A2',
    # 'EEG F4-A1',
    # 'EEG A1-A2',
     'EEG C3-A2',
    # 'EEG C4-A1',
    # 'EEG O1-A2',
    # 'EEG O2-A1',
    # 'EOG LOC-A2',
    # 'EOG ROC-A2'
]

# samples, labels = loadData(feature_file, label_file, channels)
# print("Original signal size = ", samples.shape)
# #samples, labels = loadNData(data_dir, channels, 5)
#
#
# # downsample signal from 200Hz to 50Hz
# downsample_factor = 5
# samples_down = decimate(samples, downsample_factor, axis=1)
# print("DownSampled signal size = ", samples_down.shape)
#
# plt.figure(1)
# plt.title("downsample comparison")
# plt.subplot(211)
# plt.plot(samples[0])
#
# plt.subplot(212)
# plt.plot(samples_down[0])
# plt.show()
#
#
# # fast fourier transform
# samples_fft = rfft(samples_down, axis=1)
#
# print("After fft signal size = ", samples_fft.shape)
# plt.figure(2)
# plt.subplot(211)
# plt.title("fft")
# time = range(0, 30, 0.05)
# plt.plot(time, samples_fft[0])
#
#
# # short time fourier transform
# f, t, samples_stft = stft(samples_down, fs=40, nperseg=40, noerlap=10, axis=1)
# plt.subplot(212)
# plt.pcolormesh(t, f, np.abs(samples_stft[0]))
# plt.title("STFT")
# plt.ylabel('freq [Hz]')
# plt.xlabel('time [sec]')
# plt.show()
# # print("After stft signal size = ", samples_stft.shape)


def downsample_fft(signal, downsample_factor, axis):
     """
     Downsample a signal and compute the fourier transform

     :param signal:
     :param downsample_factor:
     :param axis:
     :return:
     """
     samples_down = decimate(signal, downsample_factor, axis=axis)
     samples_fft = rfft(samples_down, axis=axis)
     return samples_fft
