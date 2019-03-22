# coding=utf-8
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM, Input, concatenate, Reshape, BatchNormalization
from keras import  backend as K
K.set_image_dim_ordering('th')
import keras


def makeConvLayers_5(inputLayer, fs):
    """
    two conv-nets in parallel for feature learning,
    one with fine resolution another with coarse resolution
    """

    # network to learn fine features
    convFine = Conv1D(filters=64, kernel_size=int(fs/2), strides=int(fs/6),
                      padding='same', activation='relu')(inputLayer)
    convFine = BatchNormalization()(convFine)
    convFine = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(convFine)
    convFine = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(convFine)
    convFine = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(convFine)
    fineShape = convFine.get_shape()
    convFine = Flatten()(convFine)

    # network to learn coarse features
    convCoarse = Conv1D(filters=32, kernel_size=fs*2, strides=int(fs/2),
                        padding='same', activation='relu')(inputLayer)
    convCoarse = BatchNormalization()(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten()(convCoarse)

    # merge
    mergeLayer = concatenate([convFine, convCoarse])
    return mergeLayer, (coarseShape, fineShape)


def makeConvLayers_8(inputLayer, fs):
    """
    two conv-nets in parallel for feature learning,
    one with fine resolution another with coarse resolution
    """

    # network to learn fine features
    convFine = Conv1D(filters=64, kernel_size=int(fs/2), strides=int(fs/6),
                      padding='same', activation='relu')(inputLayer)
    convFine = BatchNormalization()(convFine)

    convFine = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(convFine)
    convFine = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(convFine)
    convFine = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(convFine)
    convFine = BatchNormalization()(convFine)

    convFine = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(convFine)
    convFine = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(convFine)
    convFine = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(convFine)
    convFine = BatchNormalization()(convFine)

    fineShape = convFine.get_shape()
    convFine = Flatten()(convFine)

    # network to learn coarse features
    convCoarse = Conv1D(filters=32, kernel_size=fs*2, strides=int(fs/2),
                        padding='same', activation='relu')(inputLayer)
    convCoarse = BatchNormalization()(convCoarse)

    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = BatchNormalization()(convCoarse)

    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(convCoarse)
    convCoarse = BatchNormalization()(convCoarse)

    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten()(convCoarse)

    # merge
    mergeLayer = concatenate([convFine, convCoarse])
    return mergeLayer, (coarseShape, fineShape)



def preTrainingNet(n_features, n_classes, n_channels=1, fs=20):
    inLayer = Input(shape=(n_features, n_classes, n_channels))

    mLayer, (cShape, fShape) = makeConvLayers(inLayer, fs)

    outLayer = Dense(n_classes, activation='softmax')(mLayer)

    model = Model(inLayer, outLayer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


def fineTuningNet(n_features, n_classes, n_channels=1, fs=20):
    inLayer = Input(shape=(n_features, n_classes, n_channels))

    mLayer, (cShape, fShape) = makeConvLayers(inLayer, fs)

    print(mLayer)
    print(cShape)
    print(fShape)

    outLayer = Reshape((1, int(fShape[1] * fShape[2] + cShape[1] * cShape[2])))(mLayer)

    print('Reshape')
    print(outLayer)

    convLayer = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(outLayer)
    # convLayer = Dropout(rate=0.5, name='fDrop11')(convLayer)
    convLayer = BatchNormalization()(convLayer)
    convLayer = Flatten()(convLayer)

    lstmLayer = LSTM(units=128, activation='relu', dropout=0.5)(convLayer)

    mergeLayer = concatenate([convLayer, lstmLayer])

    outLayer = Dense(n_classes, activation='softmax')(mergeLayer)

    model = Model(inLayer, outLayer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
