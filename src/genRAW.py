# -*- coding: utf-8 -*-

"""
This script is used to extract features from the raw EEG signals
Features will be stored in several npz file, and all labels will be stored in corresponding npz file
You need to specify the paths to edf files and xml files, you also need to set the paths of the generated npz files
"""
import os
from preprocessing.edfLoader import extractLabels, extractRawFeature, extractEvents
from multiprocessing import Pool

if __name__ == '__main__':
    dir_path = "E:\\Document\\sleep data\\A6_data_exp_part2"
    output_file = "../data/A6_exp_data/"
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    channels = [
        # 'EEG F3-A2',
        # 'EEG F4-A1',
        # 'EEG A1-A2',
        'EEG C3-A2',
        'EEG C4-A1',
        # 'EEG O1-A2',
        # 'EEG O2-A1',
        'EOG LOC-A2',
        'EOG ROC-A2',
        'EMG Chin'
    ]

    pool = Pool() # multi process
    pool.apply_async(extractRawFeature(dir_path,output_file, channels))
    # pool.apply_async(extractLabels(dir_path,output_file))
    # pool.apply_async(extractEvents(dir_path, output_file))
    pool.close()
    pool.join()
