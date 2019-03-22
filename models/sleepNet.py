# coding=utf-8

from keras.models import Input, Model
from keras.layers import Dense, Conv1D, TimeDistributed, LSTM, BatchNormalization, Flatten, concatenate
from keras import optimizers
from keras import metrics

def build_sleepnet_lstm_model(num_features, num_classes, num_channels, timestep=3, fs=20):
    input = Input(shape=(timestep, num_features, num_channels))

    # two conv-nets in parallel for feature learning,
    # one with fine resolution another with coarse resolution
    # network to learn fine features

    # fine
    convFine = TimeDistributed(Conv1D(filters=64, kernel_size=int(fs/2), strides=int(fs/6),
                                      padding='same', activation='relu', name='fConv1'))(input)
    convFine = TimeDistributed(BatchNormalization(name='fNorm1'))(convFine)
    convFine = TimeDistributed(Conv1D(filters=64, kernel_size=8,
                                      padding='same', activation='relu', name='fConv2'))(convFine)
    convFine = TimeDistributed(Conv1D(filters=64, kernel_size=8,
                                      padding='same', activation='relu', name='fConv3'))(convFine)
    convFine = TimeDistributed(Conv1D(filters=64, kernel_size=8,
                                      padding='same', activation='relu', name='fConv4'))(convFine)
    convFine = TimeDistributed(BatchNormalization(name='fNorm2'))(convFine)
    fineShape = convFine.get_shape()
    convFine = TimeDistributed(Flatten(name='fFlat1'))(convFine)

    # coarse
    convCoarse = TimeDistributed(Conv1D(filters=32, kernel_size=fs * 2, strides=int(fs / 2),
                                        padding='same', activation='relu',name='cConv1'))(input)
    convCoarse = TimeDistributed(BatchNormalization(name='cNorm1'))(convCoarse)
    convCoarse = TimeDistributed(Conv1D(filters=64, kernel_size=6,
                                        padding='same', activation='relu', name='cConv2'))(convCoarse)
    convCoarse = TimeDistributed(Conv1D(filters=64, kernel_size=6,
                                        padding='same', activation='relu', name='cConv3'))(convCoarse)
    convCoarse = TimeDistributed(Conv1D(filters=64, kernel_size=6,
                                        padding='same', activation='relu', name='cConv4'))(convCoarse)
    convCoarse = TimeDistributed(BatchNormalization(name='cNorm2'))(convCoarse)
    coarseShape = convCoarse.get_shape()
    convCoarse = TimeDistributed(Flatten(name='cFlat1'))(convCoarse)

    x = concatenate([convFine, convCoarse], name='merge')
    x = TimeDistributed(Flatten())(x)
    # x = Flatten()(x)
    x = LSTM(32, activation='relu', name='bLstm2')(x)

    output = Dense(num_classes, activation='softmax', name='outLayer')(x)

    # adam = optimizer.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.)
    model = Model(input=input, output=output, name='sleep_lstm_1D')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[metrics.sparse_categorical_accuracy])

    return model