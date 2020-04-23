#!/usr/bin/env python3
import os

import numpy as np
from tensorflow import keras as k

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_lstm_32(frames, matrix_rows, matrix_cols, channels):
    """
    Generate 32-unit LSTM model
    """
    reg = k.regularizers.l1_l2(0.001, 0.001)
    _model = k.models.Sequential(name='asd_lstm_32', layers=[
        k.layers.Input(shape=(frames, matrix_rows, matrix_cols, channels)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        k.layers.LSTM(32, return_sequences=True, kernel_regularizer=reg, name='lstm_1'),
        k.layers.LSTM(32, kernel_regularizer=reg, name='lstm_2'),
        k.layers.Dense(64, activation='sigmoid', kernel_regularizer=reg, name='dense'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg, name='prediction')
    ])
    _model.compile(optimizer=k.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return _model


def create_conv_32(frames, matrix_rows, matrix_cols, channels):
    """
    Generate 32-unit 1D Convolution model
    """
    reg = k.regularizers.l1_l2(0.001, 0.001)
    conv_1d_spec = {
        'activation': 'relu',
        'kernel_regularizer': reg,
        'padding': 'same'
    }
    _model = k.models.Sequential(name='asd_conv_32', layers=[
        k.layers.Input(shape=(frames, matrix_rows, matrix_cols, channels)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        k.layers.Conv1D(filters=16, kernel_size=32, strides=1, **conv_1d_spec, name='conv_1'),
        k.layers.MaxPooling1D(name='pool_1'),
        k.layers.Dropout(rate=0.1, name='dropout_1'),
        k.layers.Conv1D(filters=32, kernel_size=32, strides=1, **conv_1d_spec, name='conv_2'),
        k.layers.GlobalMaxPooling1D(name='pool_2'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg, name='prediction')
    ])
    _model.compile(optimizer=k.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
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
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    print('OK')

    print('Creating Models...')
    models = [create_conv_32(*X.shape[1:]), create_lstm_32(*X.shape[1:]), ]
    print('OK')

    print('Evaluating')
    for model in models:
        model.summary()
        model.fit(X, Y, epochs=500, validation_split=0.2)
    print('Done')