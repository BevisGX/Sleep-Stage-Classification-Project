# coding=utf-8

"""
This script is used to extract features and labels from the raw EEG signals captured by Alice A6
Features are stored in .EDF files, and labels are stored in .xml(.rml) files.
Features will be stored in several npz file, and all labels will be stored in corresponding csv file
"""

import numpy as np
import pandas as pd
import pyedflib
import sys, os, re
import xml.etree.ElementTree as ET
from preprocessing.xmlTool import removePrefix, findSubelement
from preprocessing.featurePrepreocess import downsample_fft
from preprocessing.psg_preprocess import psg_preprocess

##################################
# step 1: read data from edf files
##################################
def read_edf_file(file_path, selected_channels, ecg_channel):
    """
    Read a raw edf file, and get the sampling data for the given channels

    input:
        file_path: the path to the edf file
        selected_channels: the channels to read
    output:
        eeg_raw: a k*N shape numpy array, where k is the number of channels, N=Fs*T is the total number of samples for
            one patient, Fs is the sampling rate, T is the recording duration
        labels: channel names corresponding to the signals
        Fs: sampling rate of each channel
    """
    # read the whole edf file
    if not os.path.isfile(file_path):
        raise Exception ("%s is not found" % file_path)
    print(file_path)
    edf = pyedflib.EdfReader(file_path)
    channels = edf.getSignalLabels()
    # labels = []
    eeg_raw = []
    ecg_raw = []
    fs = []
    for ii in range(len(channels)):
        if channels[ii] in selected_channels:
            # labels.append(channels[i])
            eeg_raw.append(edf.readSignal(ii))
            fs.append(edf.getSampleFrequencies()[ii])
        elif(channels[ii] == ecg_channel):
            ecg_raw = edf.readSignal(ii)

    eeg_raw = np.asarray(eeg_raw)
    return eeg_raw, fs, ecg_raw

##################################
# step 2: extract features
#################################
def extractRawFeature(dir_path, output_file, channels, ecg_channel='ECG I'):
    """
    Extract the raw eeg feature for each 30 second epoch, downsample and compute fft, and save the generated features
    to .npz file

    The feature and relative informations would be saved in .npy format and wrapped up as one .npz file

    The keys of save dictionary include:
        'channels' : a list of channel names
        'features' : a (k, n, 30*Fs/downsample_factor/2) numpy array, where k is the number of channels, Fs is the
                     sampling rate, and n is the number of epochs
        'sampling rate': a list of sampling rate related to channel names

    input:
        dir_path: the full path of the directory which contains all edf files
        output_file: the path of the output file which stores all the features
        channels: the channels to extract
    output:
        None
    """

    num_files = 0
    dirNames = os.listdir(dir_path)
    dirNames.sort()
    for dirName in dirNames:
        #if num_files == 1: break
        nsrrid = dirName.split('-')[0]
        result = []
        fnames = os.listdir(os.path.join(dir_path, dirName))
        fnames.sort()
        for fname in fnames:
            if fname.split('.')[1] != 'edf' or "T" in fname:        #bad data, sad
                continue

            psg_path = os.path.join(os.path.join(dir_path, dirName), fname)
            eeg_raw, fss, ecg_raw = read_edf_file(psg_path, channels, ecg_channel)

            fs = fss[0]         #as the sampling rates are the same
            print(eeg_raw.shape)
            k, M = eeg_raw.shape
            if M % (fs * 30) != 0:
                M = int(M/(fs*30)) * fs * 30
                eeg_raw = eeg_raw[:, 0:M]
                ecg_raw = ecg_raw[0:M]
            eeg_raw = eeg_raw.reshape((k, int(M/(30*fs)), 30*fs))
            ecg_raw = ecg_raw.reshape((int(M/(30*fs)), 30*fs))

            new_eeg_raw = []
            for ii in range(k):
                new_sig = psg_preprocess(eeg_raw[0], ecg_raw)
                new_eeg_raw.append(new_sig)

            result.append(new_eeg_raw)
        eeg_raw = result[0]
        for i in range(1, len(result)):
            eeg_raw = np.concatenate((eeg_raw, result[i]), axis = 1)        #concatenate signals orignally saved in different files to a complete signal

        eeg_raw = downsample_fft(eeg_raw, fs, axis=2)
        print("features shape = ", eeg_raw.shape)
        save_dict = {
            "channels": channels,
            "features": eeg_raw,
            "sampling rate": fss
        }
        np.savez(os.path.join(output_file, "Features_"+nsrrid), **save_dict)
        num_files += 1
    print("extracted %d files for raw features" % num_files)


#################################
# step 3: extract labels
#################################
def extractLabels(dir_path,output_file):
    """
    Extract sleep stage labels for all eeg epochs from given xml files, and output the labels to .npz file and it's
    one-hot-encoding data to .csv file

    The labels save to .npz would be in shape of (n), and in .csv file labels would be encoded by one-hot-encoding,
    the output matrix is in shape of (n, 5), where n is the number of epochs.

    input:
        dir_path: the string of directory which contains the xml files
        output_file: the string of output file path
    output:
        None
    """
    stageMap = {'NotScored': -1,
                'Wake': 0,
                'NonREM1': 1,
                'NonREM2': 2,
                'NonREM3': 3,
                'REM': 4
                }
    stageMapCsv = {-1: [1, 0, 0, 0, 0],
                   0: [1, 0, 0, 0, 0],
                   1: [0, 1, 0, 0, 0],
                   2: [0, 0, 1, 0, 0],
                   3: [0, 0, 0, 1, 0],
                   5: [0, 0, 0, 0, 1]
                   }
    num_files = 0
    dirNames = os.listdir(dir_path)
    dirNames.sort()
    for dirName in dirNames:
        nsrrid = dirName.split('-')[0]
        #if num_files == 1: break
        for fname in os.listdir(os.path.join(dir_path, dirName)):
            if fname.split('.')[1] != 'rml':
                continue
            fname = os.path.join(os.path.join(dir_path, dirName), fname)
            print(fname)
            tree = ET.parse(fname)
            root = tree.getroot()
            removePrefix(root)
            userStaging = findSubelement(root, 'UserStaging')
            duration = int(float(findSubelement(root, 'Duration').text))
            stage = []
            start = []
            for stg in userStaging.getchildren()[0].iterfind('Stage'):
                stage.append(stg.get('Type'))
                start.append(int(float(stg.get('Start'))))
            data = [-1 for i in range(int(duration/30))]
            for i in range(len(stage)-1):
                for j in range(int(start[i]/30), int(start[i+1]/30)):
                    data[j] = stageMap[stage[i]]
            for j in range(int(start[-1]/30), int(duration/30)):
                data[j] = stageMap[stage[-1]]
            data = np.asarray(data)
        save_dict = {
            "stage_labels": data
        }
        np.savez(os.path.join(output_file, "Labels_" + nsrrid), **save_dict)
        # one_hot_data = []
        # for i in range(len(data)):
        #     one_hot_data.append(stageMapCsv[data[i]])
        # np.asarray(one_hot_data)
        # one_hot_data = pd.DataFrame(one_hot_data)
        # one_hot_data.to_csv(os.path.join(output_file, "Labels_"+nsrrid+'.csv'), header=False, index=False)
        num_files += 1
    print("extracted %d files for labels" % num_files)


#################################
# step 4: extract events
#################################
def extractEvents(dir_path,output_file):
    """Extract sleep events information from .xml(rml) file

    The events will be saved to .csv file, and its columns will be ['事件', '类型', '时间', '所在帧', '持续时间', '所在信号']

    input:
        dir_path: the string of directory which contains the xml files
        output_file: the string of output file path
    output:
        None
    """
    num_files = 0
    for dirName in os.listdir(dir_path):
        nsrrid = dirName.split('-')[0]
        # if num_files == 1: break
        for fname in os.listdir(os.path.join(dir_path, dirName)):
            if fname.split('.')[1] != 'rml':
                continue
            fname = os.path.join(os.path.join(dir_path, dirName), fname)
            print(fname)
            tree = ET.parse(fname)
            root = tree.getroot()
            removePrefix(root)
            events = findSubelement(root, 'Events')
            # duration = int(float(findSubelement(root, 'Duration').text))
            data = pd.DataFrame(columns=['事件', '类型', '时间', '所在帧', '持续时间', '所在信号'])
            for event in events.iterfind('Event'):
                family = event.get('Family')
                type_ = event.get('Type')
                start = event.get('Start')
                duration = event.get('Duration')
                data = data.append({'事件': family, '类型': type_, '时间': start, '所在帧': None, '持续时间': duration, '所在信号': None}, ignore_index=True)
        data.to_csv(os.path.join(output_file, "Events_" + nsrrid + '.csv'), index=False)
        num_files += 1
    print("extracted %d files for events" % num_files)