#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
from tensorflow.keras import *


def create_model(frames, matrix_rows, matrix_cols, channels):
    """
    Generate model capable of handling multi-channel EEG time-series
    """
    conv_spec = {'kernel_size': (3, 3), 'strides': (1, 1), 'activation': 'relu', 'kernel_regularizer': regularizers.l2(0.001), 'padding': 'same'}

    _model = models.Sequential(name='asd_classifier')
    _model.add(layers.Input(shape=(frames, matrix_rows, matrix_cols, channels), name='eeg_slice'))
    _model.add(layers.TimeDistributed(layers.Conv2D(filters=32, **conv_spec), name='conv_1'))
    _model.add(layers.TimeDistributed(layers.Conv2D(filters=64, **conv_spec), name='conv_2'))
    _model.add(layers.TimeDistributed(layers.GlobalMaxPooling2D(), name='pooling'))
    _model.add(layers.LSTM(128, dropout=0.1))
    _model.add(layers.Dense(64, activation='sigmoid'))
    _model.add(layers.Dense(1, activation='sigmoid'))
    _model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return _model


def load_dataset():
    _x = ...
    _y = ...
    if os.path.exists('data/x.npy') and os.path.exists('data/y.npy'):
        print('loading npy files')
        _x, _y = np.load('data/x.npy'), np.load('data/y.npy')
        print('OK')
    else:
        print('npy files not found. creating from dataset')
        data = pd.read_feather('data/dataset-clean.ftr')
        labels = pd.DataFrame(
            data={'Label': [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]},
            index=['002', '004', '005', '007', '008', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022']
        )
        print('OK')

        # define constant values
        participants = data['Participant'].unique()  # participants
        epochs = data['Epoch'].unique()[1:]  # epochs (ignoring baseline)
        NUM_ROWS = 5
        NUM_COLS = 6
        NUM_CHANNELS = 1
        FREQ = 250  # sampling frequency

        # define parameters to extract training samples (slices)
        SLICE_SECONDS = 4  # number of seconds per slice
        SLICE_SAMPLES = FREQ * SLICE_SECONDS  # samples per slice
        SLICE_SHAPE = (SLICE_SAMPLES, NUM_ROWS, NUM_COLS, NUM_CHANNELS)

        # define x and y
        _x = np.empty(shape=(0, *SLICE_SHAPE))
        _y = np.empty(shape=(0,))

        # generate values for x and y
        print('Generating X and Y')
        data = data.set_index('Participant')
        for i, p in enumerate(participants):
            label = labels.loc[p]['Label']
            d = data.loc[p].set_index('Epoch')
            for j, e in enumerate(epochs):
                s = d.loc[e].set_index('T')  # type: pd.DataFrame
                N = len(s) // SLICE_SAMPLES  # number of possible slices
                _slices = s.iloc[:(N * SLICE_SAMPLES)].to_numpy().reshape((N, *SLICE_SHAPE))
                _x = np.append(_x, _slices, axis=0)
                _y = np.append(_y, np.full((N,), label), axis=0)
        print('OK')

        # save x and y
        print('Saving x and y')
        np.save('data/x.npy', _x)
        np.save('data/y.npy', _y)
        print('OK')
    return _x, _y


if __name__ == '__main__':
    # load dataset
    X, Y = load_dataset()
    print(f'X: {X.shape}')
    print(f'Y: {Y.shape}')

    print('Creating Model')
    model = create_model(*X.shape[1:])
    model.summary()

    print('Training')
    model.fit(X, Y, epochs=10, validation_split=0.1)
