# coding=utf-8
"""
integrate method
"""

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model


def integrate_weight(pred_y, weights):
    integrate_pred_y = (pred_y[0] * weights[0] +
                        pred_y[1] * weights[1] +
                        pred_y[2] * weights[2] +
                        pred_y[3] * weights[3] +
                        pred_y[4] * weights[4])
    integrate_pred_label = np.argmax(integrate_pred_y, axis=1)
    return integrate_pred_label


def train_integrate_nn(x, y):
    '''
    train a nn model to integrate 5 channels
    :param x:
    :param y:
    :return:
    '''
    model = Sequential()
    model.add(Dense(units=10, activation = 'relu', input_shape=(5, 6)))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x, y, validation_split=0.2)
    model.save("../model/integrate/integrate_model.h5")

def integrate_nn(pred_y):
    model = load_model("../model/integrate/integrate_model.h5")
    y = model.predict(pred_y)
    y_label = np.argmax(y)
    return y_label
