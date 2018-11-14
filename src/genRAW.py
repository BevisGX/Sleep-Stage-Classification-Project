# -*- coding: utf-8 -*-

"""
This script is used to extract features from the raw EEG signals
Features will be stored in several npz file, and all labels will be stored in corresponding npz file
You need to specify the paths to edf files and xml files, you also need to set the paths of the generated npz files
"""
import os
from preprocessing.edfLoader import extractLabels, extractRawFeature, extractEvents

if __name__ == '__main__':
    dir_path = "E:\\Document\\sleep data\\A6_data_exp_part2"
    output_file = "../data/A6_exp_data/"
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    # extractRawFeature(dir_path,output_file)
    # extractLabels(dir_path,output_file)
    extractEvents(dir_path, output_file)
