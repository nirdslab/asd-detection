#!/usr/bin/env python3
import os

import numpy as np
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


if __name__ == '__main__':
    # load dataset
    print('loading dataset...')
    data = np.load('data/data-final.npz')
    X, Y, Z = [data['x'], data['y'], data['z']]  # type: np.ndarray
    print('OK')

    print(f'X: shape={X.shape}')
    print(f'Y: shape={Y.shape}')
    print(f'Z: shape={Z.shape}')

    # normalize X
    print('normalizing data...')
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    print('OK')

    print('Creating Models...')
    models = [create_conv_32(*X.shape[1:]), create_lstm_32(*X.shape[1:]), ]
    print('OK')

    print('Evaluating')
    for model in models:
        model.summary()
        model.fit(X, Y, epochs=500, validation_split=0.2)
    print('Done')
