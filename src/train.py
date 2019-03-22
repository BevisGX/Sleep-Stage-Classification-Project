# coding=utf-8
"""
This script is used to train Sleep Stage Classfication models
"""

import os
import numpy as np
from preprocessing.npzLoader import loadNData, loadData
from preprocessing.timeDistributed import create_ngram_set
from preprocessing.sleep_stage import print_n_samples_each_class
from models import sleepNet, deepSleepNet
import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def main():
    train_dir = "../data/A6_exp_data/train_fft/"
    # test_dir = "../data/A6_exp_data/train_fft/"
    channels = [
            'EEG C3-A2',
            'EEG C4-A1',
            'EOG LOC-A2',
            'EOG ROC-A2',
            # 'EMG Chin'
        ]

    train_x, train_y = loadNData(train_dir, channels)
    #test_x, test_y = loadNData(test_dir, channels, include=test_id)

    n_classes = 6
    n_channels = len(channels)
    n_features = train_x.shape[1]
    train_x = np.reshape(train_x, (train_x.shape[0],n_features, n_channels))
    # test_x = np.reshape(test_x, (test_x.shape[0],n_features, n_channels))

    time_step = 3
    train_x = create_ngram_set(train_x, num_gram=time_step)
    # test_x = create_ngram_set(test_x, num_gram=time_step)

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, )

    print("train_x: ", train_x.shape)
    print("train_y: ", train_y.shape)
    print_n_samples_each_class(train_y)
    print("test_x: ", test_x.shape)
    print("test_y: ", test_y.shape)


    # model
    model = sleepNet.build_sleepnet_lstm_model(n_features, n_classes, n_channels, time_step)
    # model = deepSleepNet.fineTuningNet(n_features, n_classes, n_channels)

    model.summary()

    save_dir = "../models/4_channels_sleepNet"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # callbacks
    stopping = keras.callbacks.EarlyStopping(patience=8)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                  patience=2,
                                                  min_lr=1e-6)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, "{val_loss:.3f}-{val_sparse_categorical_accuracy:.3f}-{epoch:03d}-{loss:.3f}-{sparse_categorical_accuracy:.3f}.hdf5"),
        save_best_only=False

    )

    batch_size = 32
    class_weight = compute_class_weight('balanced', np.unique(train_y), train_y)

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=100,
                        shuffle=True,
                        class_weight=class_weight,
                        validation_data=(test_x, test_y),
                        callbacks=[
                            reduce_lr,
                            checkpoint,
                            stopping
                        ])

    # evaluate
    pred_train_y = model.predict(train_x)
    pred_train_label = np.argmax(pred_train_y, axis=1)
    train_cm = confusion_matrix(train_y, pred_train_label)
    print(train_cm)

    pred_test_y = model.predict(test_x)
    pred_test_label = np.argmax(pred_test_y, axis=1)

    valid_cm = confusion_matrix(test_y, pred_test_label)
    print(valid_cm)

    scores = accuracy_score(test_y, pred_test_label)
    print('Test accuracy:', scores)

    target_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    report = classification_report(test_y, pred_test_label, target_names=target_names)
    print(report)


    # history_dir = os.path.join(train_dir, "history/")
    # if not os.path.exists(history_dir):
    #     os.makedirs(history_dir)
    # np.save(os.path.join(history_dir, "Pred_" + test_id.split('.')[0]), pred_test_label)
    # with open (os.path.join(history_dir, 'test_sample_{}'.format(test_id)), 'w') as history_file:
    #     print(valid_cm, file=history_file)
    #     print(report, file=history_file)
    #     print('Test accuracy:{}'.format(scores), file=history_file)
    #     print(history['acc'], file=history_file)
    #     print(history['loss'], file=history_file)
    #     print(history['val_acc'], file=history_file)
    #     print(history['val_loss'], file=history_file)



# data_dir = "../data/A6_exp_data/train_fft/"
# fnames = os.listdir(data_dir)
# fnames.sort()
# for i in range(len(fnames)):
#     if fnames[i].split('_')[0] == 'Labels':
#         continue
#     test_id = fnames[i].split('_')[1]
#     test(test_id)
if __name__ == main():
    main()