# -*- coding: utf-8 -*-
"""
Created on 2018-03-10
Trains neural network on Beijing TongRen Hospital training datasets

@author: Huang Haiping
"""

import win_unicode_console
win_unicode_console.enable()

import os
import numpy as np
from models import deepSleepNet as dsn
import keras
from preprocessing.data_loader import NonSeqDataLoader, SeqDataLoader
from sklearn.metrics import confusion_matrix
from preprocessing.utils import plot_confusion_matrix

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Prepare model model saving directory.
#save_dir = os.path.join(os.getcwd(), '/models/saved_models_tongren_deepsleepnet')
save_dir = "../models/saved_models_tongren_deepsleepnet"
model_name = 'sleep_s_model.{epoch:03d}.h5' 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
# =============================================================================
# Callbacks
# =============================================================================
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

history = AccuracyHistory()

callbacks = [checkpoint, lr_reducer, lr_scheduler, history]   
       
# =============================================================================
# Script
# =============================================================================
# Data
data_dir = "datasets/tsinghua/EMG_Chin_FFT_1024"
n_folds = 10
fold_idx = 1


n_classes=5
epochs = 10
num_gram = 0

data_prefix = "../data/tsinghua/"
data_dirs = [
             data_prefix + "Train",
             #data_prefix + "EEG_C3-A2_Feature_1",
             #data_prefix + "EOG_Left_Feature_1", 
             #data_prefix + "EOG_Right_Feature_1",
             #data_prefix + "EMG_Chin_Feature_1",
             #data_prefix + "EEG_C4-A1_FFT_3",
             #data_prefix + "EEG_C3-A2_FFT",
             #data_prefix + "EOG_Left_FFT", 
             #data_prefix + "EOG_Right_FFT",
             #data_prefix + "EMG_Chin_FFT",
             #data_prefix + "EEG_C4-A1_Wavelet",
             #data_prefix + "EEG_C3-A2_Wavelet",
             #data_prefix + "EOG_Left_Wavelet", 
             #data_prefix + "EOG_Right_Wavelet",
             #data_prefix + "EMG_Chin_Wavelet",
             #data_prefix + "EOG_Right_Raw"
             #data_prefix + "EEG_C4-A1_Raw",
             ]
#n_feats=1024 * (len(data_dirs))

#Loading datasets
data_loader = NonSeqDataLoader(
    data_dir=data_dir, 
    n_folds=n_folds, 
    fold_idx=fold_idx,
    data_dirs = data_dirs
)
x_train, y_train = data_loader.load_train_data_set(Oversample=False, Downsample=False, num_gram=num_gram)

#x_train = x_train/255

print(type(x_train))
print(x_train.shape)
print(y_train.shape)
            
n_feats = x_train.shape[1]

# reshape to keep input to NN consistent
X_train_ovs = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# pre-training phase
preTrain = dsn.preTrainingNet(n_feats, n_classes)



data_prefix = "../data/tsinghua/"
data_dirs = [
             data_prefix + "Test",
             #data_prefix + "EEG_C3-A2_Feature_1",
             #data_prefix + "EOG_Left_Feature_1", 
             #data_prefix + "EOG_Right_Feature_1",
             #data_prefix + "EMG_Chin_Feature_1",
             #data_prefix + "EEG_C4-A1_FFT_3",
             #data_prefix + "EEG_C3-A2_FFT",
             #data_prefix + "EOG_Left_FFT", 
             #data_prefix + "EOG_Right_FFT",
             #data_prefix + "EMG_Chin_FFT",
             #data_prefix + "EEG_C4-A1_Wavelet",
             #data_prefix + "EEG_C3-A2_Wavelet",
             #data_prefix + "EOG_Left_Wavelet", 
             #data_prefix + "EOG_Right_Wavelet",
             #data_prefix + "EMG_Chin_Wavelet",
             #data_prefix + "EOG_Right_Raw",
             #data_prefix + "EEG_C4-A1_Raw",
             ]
#Loading datasets
data_loader = NonSeqDataLoader(
    data_dir=data_dir, 
    n_folds=n_folds, 
    fold_idx=fold_idx,
    data_dirs = data_dirs
)
x_test, y_test = data_loader.load_train_data_set(Oversample=False, Downsample=False, num_gram=num_gram)


#x_test = x_test/255

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)


# training this network on oversampled dataset
preTrain.fit(X_train_ovs, 
             y_train, 
             epochs=epochs, 
             batch_size=100,  
             shuffle=True,
             class_weight=class_weights, 
             validation_data=(x_test, y_test),
             callbacks=callbacks)


score = preTrain.evaluate(X_train_ovs, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

pred_train_y = preTrain.predict(X_train_ovs)
pred_train_label = np.argmax(pred_train_y, axis=1)
train_cm = confusion_matrix(y_train, pred_train_label)
print(train_cm)



score = preTrain.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred_test_y = preTrain.predict(x_test)
pred_test_label = np.argmax(pred_test_y, axis=1)
valid_cm = confusion_matrix(y_test, pred_test_label)
print(valid_cm)


print(np.transpose(history.acc))
print(np.transpose(history.val_acc))


# save neural network weights so that we can use them while testing    
#preTrain.save_weights('supervisePreTrainNet_TestSub'+ str(fold_idx) +'.h5')


