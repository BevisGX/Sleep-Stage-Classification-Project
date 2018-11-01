# coding=utf-8

"""
This script is used to extract features from the raw EEG signals
Features will be stored in several npz file, and all labels will be stored in corresponding npz file
"""

import numpy as np
import pyedflib
import sys, os, re
import xml.etree.ElementTree as ET
from preprocessing.xmlTool import removePrefix, findSubelement

##################################
# step 1: read data from edf files
##################################
def read_edf_file(file_path):
    """
    read a raw edf file, and get the sampling data for the given channels
    input:
        file_path: the path to the edf file
    output:
        eeg_raw: a k*N shape numpy array, where k (k=2) is the number of channels, N=Fs*T is the total number of samples for one patient, Fs is the sampling rate, T is the recording duration
        labels: channel names corresponding to the signals
        Fs: sampling rate of each channel
    """
    # read the whole edf file
    if not os.path.isfile(file_path):
        raise Exception ("%s is not found" % file_path)
    print(file_path)
    edf = pyedflib.EdfReader(file_path)
    channels = edf.getSignalLabels()
    labels = []
    eeg_raw = []
    fs = []
    for i in range(len(channels)):
        if "EEG" in channels[i] or "EOG" in channels[i] or "EMG" in channels:
            labels.append(channels[i])
            eeg_raw.append(edf.readSignal(i))
            fs.append(edf.getSampleFrequencies()[i])
    eeg_raw = np.asarray(eeg_raw)
    return eeg_raw, labels, fs

##################################
# step 2: extract features
#################################
def extractRawFeature(dir_path, output_file):
    """
    extract the raw eeg feature for each 30 second epoch, and output the generated features to a text file
    The feature array outputed will be in shape of k*n*30Fs, where k is the number of channels, Fs is the samping rate, and n is the number of epoches. The feature arrays for all patients will be outputted to a npz file
    input:
        dir_path: the full path of the directory which contains all edf files
        output_file: the path of the output file which stores all the features
    output:
        None
    """

    num_files = 0
    for dirName in os.listdir(dir_path):
        #if num_files == 1: break
        nsrrid = dirName.split('-')[0]
        result = []
        for fname in os.listdir(dir_path+'/'+dirName):
            if fname.split('.')[1] != 'edf' or "T" in fname:        #bad data, sad
                continue
            fname = dir_path+'/'+dirName+'/'+fname
            eeg_raw, labels, fss = read_edf_file(fname)
            fs = fss[0]         #as the sampling rates are the same
            print(eeg_raw.shape)
            k, M = eeg_raw.shape
            if M % (fs * 30) != 0:
                M = int(M/(fs*30)) * fs * 30
                eeg_raw = eeg_raw[:, 0:M]
            eeg_raw = eeg_raw.reshape((k, int(M/(30*fs)), 30*fs))
            result.append(eeg_raw)
        eeg_raw = result[0]
        for i in range(1, len(result)):
            eeg_raw = np.concatenate((eeg_raw, result[i]), axis = 1)        #concatenate signals orignally saved in different files to a complete signal
        save_dict = {
            "channels": labels,
            "eeg_raw": eeg_raw,
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
    extract labels for all eeg epoches from given xml files, and output the labels to text file
    The output matrix is in shape of N*n, where N is the number of patient, n is the number of epoches.
    input:
        dir_path: the directory which contains the xml files
        output_file: the outputted file
    output:
        None
    """
    stageMap = {'NotScored': -1,
                'Wake': 0,
                'NonREM1': 1,
                'NonREM2': 2,
                'NonREM3': 3,
                'REM': 5
                }
    result = []
    row_indexer = 0
    num_files = 0
    for dirName in os.listdir(dir_path):
        nsrrid = dirName.split('-')[0]
        #if num_files == 1: break
        for fname in os.listdir(dir_path+'/'+dirName):
            if fname.split('.')[1] != 'rml':
                continue
            fname = dir_path+'/'+dirName+'/'+fname
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
        np.savez(os.path.join(output_file, "Labels_"+nsrrid), **save_dict)
        num_files += 1
    print("extracted %d files for labels" % num_files)