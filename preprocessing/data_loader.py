import os

import numpy as np

from preprocessing.sleep_stage import print_n_samples_each_class
from preprocessing.utils import get_balance_class_oversample,get_balance_class_downsample,get_balance_class_downsample_2, get_balance_class_downsample_3, get_balance_class_number_sample
from sklearn.preprocessing import MinMaxScaler

import re
from training import trainer_tools


class NonSeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx, data_dirs=None):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.data_dirs = data_dirs

    def _load_npz_file(self, npz_file):
        """Load datasets and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        print("data: ", data.shape, " labels: ", labels.shape)
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load datasets and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        begin = 0;
        end = 0;
        for npz_f in npz_files:
            #print ("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            end = end + len(tmp_data)
            '''
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            '''
            data.append(tmp_data)
            labels.append(tmp_labels)
            #print(npz_f, "Begin: ", begin, " End: ", end)
            begin = end
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def _load_cv_data(self, list_files):
        """Load training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print (" ")
        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print (" ")

        # Reshape the datasets to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files=None, Oversample=True):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))

        if len(subject_files) == 0:
            for idx, f in enumerate(allfiles):
                if self.fold_idx < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
                if pattern.match(f):
                    subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print (" ")
        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
        print (" ")

        # Reshape the datasets to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        print ("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train)
        print (" ")
        print ("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        print_n_samples_each_class(label_val)
        print (" ")
        
        
        if(Oversample==False):
            return data_train, label_train, data_val, label_val
        
        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print ("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print (" ")

        return x_train, y_train, data_val, label_val
    
    
    def load_train_data_2(self, n_files=None, Oversample=True):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]
            
        
        '''
        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))

        if len(subject_files) == 0:
            for idx, f in enumerate(allfiles):
                if self.fold_idx < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
                if pattern.match(f):
                    subject_files.append(os.path.join(self.data_dir, f))
        '''

        train_files = list(set(npzfiles))
        train_files.sort()
        #subject_files.sort()

        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(data_train.shape)
        print(label_train.shape)
        print (" ")
        #print ("Load validation set:")
        #data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
        #print (" ")
        
        #data_train = MinMaxScaler().fit_transform(data_train)
        
        #from sklearn.feature_selection import SelectFromModel
        #from sklearn.linear_model import LogisticRegression
        
        #带L1惩罚项的逻辑回归作为基模型的特征选择
        #data_train = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(data_train, label_train)

        # Reshape the data to mdatasets the input of the model - conv2d
        data_train = np.squeeze(data_train)
        #data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        #data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        #data_val = data_val.astype(np.float32)
        #label_val = label_val.astype(np.int32)

        print ("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train)
        print (" ")
        #print ("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        #print_n_samples_each_class(label_val)
        #print (" ")
        
        
        if(Oversample==False):
            return data_train, label_train
        
        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_downsample_2(
            x=data_train, y=label_train
        )
        print ("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print (" ")

        return x_train, y_train
    
    
    def load_train_data_3(self, n_files=None, Oversample=True):
        # Remove non-mat files, and perform ascending sort
        
        data_trains = []
        label_trains = []
        for data_dir in self.data_dirs:
            allfiles = os.listdir(data_dir)
            npzfiles = []
            for idx, f in enumerate(allfiles):
                if ".npz" in f:
                    npzfiles.append(os.path.join(data_dir, f))
            npzfiles.sort()
    
            if n_files is not None:
                npzfiles = npzfiles[:n_files]
                
            train_files = list(set(npzfiles))
            train_files.sort()
            #subject_files.sort()
            data_train, label_train = self._load_npz_list_files(npz_files=train_files)
            print(data_train.shape, " ", len(data_train))
            print(label_train.shape)
            print (" ")
            for ii in range(len(data_train)):  
                           
                if(len(data_trains) < len(data_train)):
                    values = []
                    counter = 0
                    for value in data_train[ii]:
                        #print("value: ", type(value), " ", value.shape, " ", value , " " , counter)
                        #if(counter > 5):
                        #    break
                        values.append(value)
                        counter = counter + 1
                    data_trains.append(values)
                    label_trains.append(label_train[ii])
                else:
                    counter = 0
                    for value in data_train[ii]:
                        #if(counter > 5):
                        #    break
                        data_trains[ii].append(value)
                        counter = counter + 1
            
            
        #print(label_trains.shape)
        data_trains = np.array(data_trains)
        label_trains = np.array(label_trains)
        
        print(data_trains.shape)
        print(label_trains.shape)
        print (" ")
        
        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        #data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(data_trains.shape)
        print(label_trains.shape)
        print (" ")

        # Casting
        data_trains = data_trains.astype(np.float32)
        label_trains = label_trains.astype(np.int32)

        print ("Training set: {}, {}".format(data_trains.shape, label_trains.shape))
        print_n_samples_each_class(label_train)
        print (" ")

        
        if(Oversample==False):
            return data_trains, label_trains
        
        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_downsample_3(
            x=data_trains, y=label_trains
        )
        print ("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print (" ")

        return x_train, y_train
    
    def load_train_data_set(self, n_files=None, Oversample=False, Downsample=False, num_gram=0, samples=0):
        # Remove non-mat files, and perform ascending sort
        
        data_trains = []
        label_trains = []
        for data_dir in self.data_dirs:
            allfiles = os.listdir(data_dir)
            npzfiles = []
            for idx, f in enumerate(allfiles):
                if ".npz" in f:
                    npzfiles.append(os.path.join(data_dir, f))
            npzfiles.sort()
    
            if n_files is not None:
                npzfiles = npzfiles[:n_files]
                
            train_files = list(set(npzfiles))
            train_files.sort()
            #subject_files.sort()
            data_train, label_train = self._load_npz_list_files(npz_files=train_files)
            
            if(num_gram > 0):
                data_train = trainer_tools.create_ngram_set(data_train, num_gram)
            
            print(data_train.shape, " ", len(data_train))
            print(label_train.shape)
              
            if len(data_trains) > 0 :
                data_trains = np.hstack([data_trains, data_train])
            else:
                data_trains = data_train
            
            
        #data_trains = np.vstack([np.array(data_trains)])
        #print(label_trains.shape)
        #data_trains = np.array(data_trains)
        label_trains = np.array(label_train)
        
        print(data_trains.shape)
        print(label_trains.shape)
        print (" ")
        
        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        #data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(data_trains.shape)
        print(label_trains.shape)
        print (" ")

        # Casting
        data_trains = data_trains.astype(np.float32)
        label_trains = label_trains.astype(np.int32)

        print ("Training set: {}, {}".format(data_trains.shape, label_trains.shape))
        print_n_samples_each_class(label_train)
        print (" ")

        
        if(Oversample==True):
            # Use balanced-class, oversample training set
            x_train, y_train = get_balance_class_oversample(
                x=data_trains, y=label_trains
            )
            print ("Oversampled training set: {}, {}".format(
                x_train.shape, y_train.shape
            ))
            print_n_samples_each_class(y_train)
            print (" ")
        elif(Downsample == True):
            # Use balanced-class, oversample training set
            x_train, y_train = get_balance_class_downsample(
                x=data_trains, y=label_trains
            )
            print ("Downsampled training set: {}, {}".format(
                x_train.shape, y_train.shape
            ))
            print_n_samples_each_class(y_train)
            print (" ")
        elif(samples > 0):
            # Use balanced-class, oversample training set
            x_train, y_train = get_balance_class_number_sample(
                x=data_trains, y=label_trains, samples = samples
            )
            print ("Sampled training set: {}, {}".format(
                x_train.shape, y_train.shape
            ))
            print_n_samples_each_class(y_train)
            print (" ")
        
        else:
            return data_trains, label_trains
        
        return x_train, y_train
    
    
    def load_train_data_set_2(self, n_files=None, Oversample=False, Downsample=False, num_gram=0):
        # Remove non-mat files, and perform ascending sort
        
        data_trains = []
        label_trains = []
        for data_dir in self.data_dirs:
            allfiles = os.listdir(data_dir)
            npzfiles = []
            for idx, f in enumerate(allfiles):
                if ".npz" in f:
                    npzfiles.append(os.path.join(data_dir, f))
            npzfiles.sort()
    
            if n_files is not None:
                npzfiles = npzfiles[:n_files]
                
            train_files = list(set(npzfiles))
            train_files.sort()
            #subject_files.sort()
            data_train, label_train = self._load_npz_list_files(npz_files=train_files)
            
            '''
            if(num_gram > 0):
                data_train = trainer_tools.create_ngram_set(data_train, num_gram)
                
            if len(data_trains) > 0 :
                data_trains = np.hstack([data_trains, data_train])
            else:
                data_trains = data_train
            '''
               
            data_trains.append(data_train)
            print(data_train.shape, " ", len(data_train))
            print(label_train.shape)
            
        #data_trains = np.vstack([np.array(data_trains)])
        #print(label_trains.shape)
        data_trains = np.array(data_trains)
        label_trains = np.array(label_train)
        
        print(data_trains.shape)
        print(label_trains.shape)
        print (" ")
        
        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        #data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(data_trains.shape)
        print(label_trains.shape)
        print (" ")

        # Casting
        data_trains = data_trains.astype(np.float32)
        label_trains = label_trains.astype(np.int32)

        print ("Training set: {}, {}".format(data_trains.shape, label_trains.shape))
        print_n_samples_each_class(label_train)
        print (" ")

        
        if(Oversample==True):
            # Use balanced-class, oversample training set
            x_train, y_train = get_balance_class_oversample(
                x=data_trains, y=label_trains
            )
            print ("Oversampled training set: {}, {}".format(
                x_train.shape, y_train.shape
            ))
            print_n_samples_each_class(y_train)
            print (" ")
        elif(Downsample == True):
            # Use balanced-class, oversample training set
            x_train, y_train = get_balance_class_downsample(
                x=data_trains, y=label_trains
            )
            print ("Downsampled training set: {}, {}".format(
                x_train.shape, y_train.shape
            ))
            print_n_samples_each_class(y_train)
            print (" ")
        else:
            return data_trains, label_trains
        
        return x_train, y_train

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))
        subject_files.sort()

        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))

        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(subject_files)

        # Reshape the data to mdatasets the input of the model
        data_val = np.squeeze(data_val)
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_val, label_val


class SeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and datasetsls from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and datasetsls from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print ("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to mdatasets the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
            
            # # Reshape the data to mdatasets the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_cv_data(self, list_files):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print (" ")
        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print (" ")

        return data_train, label_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        # Files for validation sets
        val_files = np.array_split(npzfiles, self.n_folds)
        val_files = val_files[self.fold_idx]

        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))

        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files)

        return data_val, label_val

    def load_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print (" ")
        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(subject_files)
        print (" ")

        print ("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            print (d.shape)
            n_train_examples += d.shape[0]
        print ("Number of examples = {}".format(n_train_examples))
        print_n_samples_each_class(np.hstack(label_train))
        print (" ")
        print ("Validation set: n_subjects={}".format(len(data_val)))
        n_valid_examples = 0
        for d in data_val:
            print (d.shape)
            n_valid_examples += d.shape[0]
        print ("Number of examples = {}".format(n_valid_examples))
        print_n_samples_each_class(np.hstack(label_val))
        print (" ")
        
        # Reshape the data to mdatasets the input of the model - conv2d
        #data_train = np.squeeze(data_train)
        #data_val = np.squeeze(data_val)
        #data_train = data_train[:, :, np.newaxis, np.newaxis]
        #data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        #data_val = data_val.astype(np.float32)
        #label_val = label_val.astype(np.int32)
        
        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print ("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print (" ")

        return x_train, y_train, data_val, label_val

    @staticmethod
    def load_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        subject_files = []
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(subject_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(subject_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, f))

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data and datasetsls from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and datasetsls from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print ("Loading {} ...".format(npz_f))
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to mdatasets the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
                
                # # Reshape the data to mdatasets the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        print ("Load data fromdatasets".format(subject_files))
        data, labels = load_npz_list_files(subject_files)

        return data, labels
