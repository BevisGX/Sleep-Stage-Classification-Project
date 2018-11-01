# -*- coding: utf-8 -*-

'''
Created on 2018年8月30日

@author: Huang Haiping
'''
import argparse
import glob
import ntpath
import os
import shutil
import numpy as np
from multiprocessing import Pool
from scipy import signal

# =============================================================================
# Log
# =============================================================================
# 2018-04-02 改为多线程执行
#

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

# Stages
stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

# Annotation to label
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}


def read_psg_csv(psg_file, output_dir, data_type, channel,  numBands=128, sfreq=100):
    
    """
    Read PSG from CSV file  读取PSG信息
    
    Parameters
    ----------
    psg_file : psf 文件名
    ann_file : 标签文件名
    output_dir : 数据转换后，进行保存的文件名
    data_type: 数据类型， raw(裸文件，EDF是什么内容，就读取什么内容), 
        raw: 裸文件，EDF是什么内容，就读取什么内容，然后压缩长度
        fft: 进行快速傅里叶转换，取前50Hz的信息
        feature: 进行快速傅里叶转换，取delta、theta、alpha、beta、gamma、spindle
        wavelet: 进行小波转换
    channel: 信号类型 eeg, eog, emg
    numBands: 波段长度   
    """
    
    # Data of PSG
    psg = np.loadtxt(psg_file, delimiter=',')
    # Labels
    #anns = np.loadtxt(ann_file, delimiter=',')
    
    print(psg.shape)
    #print(anns.shape)
    
    x = psg[:,1:5001].astype(np.float16)  
    y = psg[:,0].astype(np.int)
    '''
    x = np.fft.fft(x)
    
    new_x = []
    
    for value in x:
        band1 = np.fft.fft(value[0:100])
        band2 = np.fft.fft(value[100:200])
        band = []
        for b in band1:
            band.append(b.real)
        for b in band2:
            band.append(b.real)   
        new_x.append(np.array(band))
    
    x = np.array(new_x)
    x = x.astype(np.float16)
    '''
    new_x = []
    new_y = []
    feature_len = 1000
    frequencyPerFrame = 100
    for ii in range(len(x)):
        value = x[ii] * 100
        #value = _bandpass(value)
        eeg_c3 = value[0:feature_len]
        #eeg_c3 = (eeg_c3 - np.mean(eeg_c3)) / np.std(eeg_c3)
        #eeg_c3 = abs(np.fft.fft(eeg_c3))
        eeg_c3 = short_fft(eeg_c3, frequencyPerFrame)
        
        eeg_c4 = value[feature_len:feature_len * 2]
        #eeg_c4 = (eeg_c4 - np.mean(eeg_c4)) / np.std(eeg_c4)
        #eeg_c4 = abs(np.fft.fft(eeg_c4))
        eeg_c4 = short_fft(eeg_c4, frequencyPerFrame)
        
        eog_left = value[feature_len * 2:feature_len * 3]
        #eog_left = (eog_left - np.mean(eog_left)) / np.std(eog_left)
        #eog_left = abs(np.fft.fft(eog_left))
        eog_left = short_fft(eog_left, frequencyPerFrame)
        
        eog_right = value[feature_len * 3:feature_len * 4]
        #eog_right = (eog_right - np.mean(eog_right)) / np.std(eog_right)
        #eog_right = abs(np.fft.fft(eog_right))
        eog_right = short_fft(eog_right, frequencyPerFrame)
        
        emg_chin = value[feature_len * 4:feature_len * 5]
        #emg_chin = (emg_chin - np.mean(emg_chin)) / np.std(emg_chin)
        #emg_chin = abs(np.fft.fft(emg_chin))
        emg_chin = short_fft(emg_chin, frequencyPerFrame)
        
        target = y[ii]
        '''
        if target == 2 or target == 3:
            target = 1
        
        if target == 4:
            target = 2
        '''
        if target > 4:
            target = 0
        
        
        band = np.hstack([eeg_c3, eeg_c4, eog_left, eog_right, emg_chin])   
        new_x.append(np.array(band))
        new_y.append(target)
            
    x = np.array(new_x)    
    y = np.array(new_y)
    
    print("Result:")
    print(x.shape)
    print(y.shape)
    print(" ")
    # Save
    filename = ntpath.basename(psg_file).replace(".csv", ".npz")
    save_dict = {
        "x": x, 
        "y": y, 
        "fs": sfreq,
        "ch_label": channel,
    }
    np.savez(os.path.join(output_dir, filename), **save_dict)
    print("\n=======================================\n")


def _bandpass(data, fc_low = 1, fc_high = 50, fre = 100, order = 2): 
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

def short_fft(value, frequency):
    length = (int)(len(value) / frequency)
    #print("length: ", length)
    bands = np.zeros(frequency)
    max_bands = np.zeros(frequency)
    all_bands = []
    energy = []
    max = -1
    for ii in range(length):
        x = value[ii * frequency : (ii+1) * frequency]
        fft_value = abs(np.fft.fft(x))
        bands = bands + fft_value
        
        sum_bands = np.sum(bands)
        energy.append(sum_bands)
        if  sum_bands > max:
            max = sum_bands
            max_bands = bands

        all_bands.append(fft_value)
    energy.append(max)
    all_bands.append(bands)
    all_bands.append(max_bands)
    all_bands.append(np.array(energy))
    return np.hstack(all_bands)

def compressFrequence(x, numBands=256):
    ff = x
    N = len(x)

    dataPointsPerBand =int( N/numBands) 
    bands = [] # create an array to hold all the bands
    for x in range(0, numBands):
        bandValue = 0.0
        counter = 0
        for i in range(x * dataPointsPerBand, (x + 1) * dataPointsPerBand):
            if(i < N):
                bandValue += abs(ff[i].real)
                counter = counter + 1
        if(counter > 0):
            bandValue /= counter
            bands.append(bandValue)
    return np.array(bands)        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="c:/Users/Mingkai/Documents/睡眠医疗/data/A6_14_Sleep",
                        help="File path to the CSV or NPY file that contains walking datasets.")
    parser.add_argument("--output_dir", type=str, default="../data/tsinghua/EEG_C3-A2_FFT_1024",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG_C3-A2",
                        help="File path to the trained model used to estimate walking speeds.")   
    parser.add_argument("--band_len", type=int, default=256,                           
                        help="Band length of FFT.") 
    
    parser.add_argument("--data_type", type=str, default="feature",
                        help="preprocessing data, include raw data, fft, wavelet, feature.")  
    
    parser.add_argument("--channel", type=str, default="eeg",
                        help="Name of channels: eeg, eog, emg")
    
    args = parser.parse_args()
    
    print("output_dir: ", args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channel
    select_ch = args.select_ch 
    numBands = args.band_len
    data_type = args.data_type
    channel = args.channel
    
    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, str("*.csv")))
    #ann_fnames = glob.glob(os.path.join(args.data_dir, "*SleepStage_Alice5.csv"))
    psg_fnames.sort()
    #ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    #ann_fnames = np.asarray(ann_fnames)  
    print(psg_fnames)
    
    
    pool = Pool()
    for ii in range(len(psg_fnames)):
        pool.apply_async(read_psg_csv, args=(psg_fnames[ii],args.output_dir, data_type, channel, numBands))
    
    pool.close()
    pool.join()
    



if __name__ == "__main__":
    main()