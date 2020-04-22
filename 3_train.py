#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
from tensorflow import keras as k

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_lstm_32(frames, matrix_rows, matrix_cols, channels):
    """
    Generate 32-unit LSTM model
    """
    _model = k.models.Sequential(name='asd_lstm_32', layers=[
        k.layers.Input(shape=(frames, matrix_rows, matrix_cols, channels)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        k.layers.LSTM(32, return_sequences=True, kernel_regularizer='l2', name='lstm_1'),
        k.layers.LSTM(32, kernel_regularizer='l2', name='lstm_2'),
        k.layers.Dense(64, activation='sigmoid', kernel_regularizer='l2', name='dense'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='prediction')
    ])
    _model.compile(optimizer=k.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return _model


def create_conv_32(frames, matrix_rows, matrix_cols, channels):
    """
    Generate 32-unit 1D Convolution model
    """
    conv_1d_spec = {
        'kernel_size': 250,
        'strides': 2,
        'activation': 'relu',
        'kernel_regularizer': 'l2',
        'padding': 'same'
    }
    _model = k.models.Sequential(name='asd_conv_32', layers=[
        k.layers.Input(shape=(frames, matrix_rows, matrix_cols, channels)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        k.layers.Conv1D(filters=32, **conv_1d_spec, name='conv_1d'),
        k.layers.GlobalMaxPooling1D(name='pool_1d'),
        k.layers.Dense(64, activation='sigmoid', kernel_regularizer='l2', name='dense'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='prediction')
    ])
    _model.compile(optimizer=k.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return _model


def load_dataset():
    _x = ...
    _y = ...
    _z = ...
    if os.path.exists('data/dataset.npz'):
        print('loading npz dataset')
        _data = np.load('data/dataset.npz')
        _x, _y, _z = [_data['x'], _data['y'], _data['z']]
        print('OK')
    else:
        print('npy files not found. creating from dataset')
        data = pd.read_feather('data/dataset-clean.ftr')
        labels = pd.DataFrame(
            data={
                'Label': [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                'Score': [19.0, 12.0, 5.0, 0.0, 5.0, 11.0, 16.0, 16.0, 0.0, 7.0, 4.0, 0.0, 20.0, 2.0, 9.0, 4.0, 0.0]
            },
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

        # define x, y, and z
        _x = np.empty(shape=(0, *SLICE_SHAPE))  # sample
        _y = np.empty(shape=(0,))  # label
        _z = np.empty(shape=(0,))  # ADOS-2 score

        # generate values for x, y, and z
        print('Generating X, Y, and Z')
        data = data.set_index('Participant')
        for i, p in enumerate(participants):
            label = labels.loc[p]['Label']
            score = labels.loc[p]['Score']
            d = data.loc[p].set_index('Epoch')
            for j, e in enumerate(epochs):
                s = d.loc[e].set_index('T')  # type: pd.DataFrame
                N = len(s) // SLICE_SAMPLES  # number of possible slices
                _slices = s.iloc[:(N * SLICE_SAMPLES)].to_numpy().reshape((N, *SLICE_SHAPE))
                _x = np.append(_x, _slices, axis=0)
                _y = np.append(_y, np.full((N,), label), axis=0)
                _z = np.append(_z, np.full((N,), score), axis=0)
        print('OK')

        # save x, y, and z
        print('Saving x, y, and z')
        np.savez_compressed('data/dataset.npz', x=_x, y=_y, z=_z)
        print('OK')
    return _x, _y, _z


if __name__ == '__main__':
    # load dataset
    X, Y, Z = load_dataset()
    print(f'X: {X.shape}')
    print(f'Y: {Y.shape}')
    print(f'Z: {Z.shape}')

    # normalize X
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    print('Creating Models')
    models = [create_conv_32(*X.shape[1:]), create_lstm_32(*X.shape[1:]), ]

    print('Evaluating')
    for model in models:
        model.summary()
        model.fit(X, Y, epochs=500, validation_split=0.25)
    print('Done')
