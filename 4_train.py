#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras as k

tf.random.set_seed(42)


def create_lstm_model(frames, ch_rows, ch_cols, bands):
    """
    Generate LSTM model
    """
    return k.models.Sequential(name='asd_lstm_32', layers=[
        # inout
        k.layers.Input(shape=(frames, ch_rows, ch_cols, bands)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        # lstm 1
        k.layers.LSTM(32, return_sequences=True, dropout=0.2, name='lstm_1'),
        # lstm 2
        k.layers.LSTM(64, dropout=0.2, name='lstm_2'),
        # prediction
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])


def create_conv_model(frames, ch_rows, ch_cols, bands):
    """
    Generate 1D Convolution model
    """
    return k.models.Sequential(name='asd_conv_32', layers=[
        # input
        k.layers.Input(shape=(frames, ch_rows, ch_cols, bands)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        # convolution 1
        k.layers.Conv1D(filters=64, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_1'),
        k.layers.MaxPooling1D(name='pool_1'),
        k.layers.Dropout(0.2, name='dropout_1'),
        k.layers.BatchNormalization(name='batch_norm_1'),
        # convolution 2
        k.layers.Conv1D(filters=128, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_2'),
        k.layers.MaxPooling1D(name='pool_2'),
        k.layers.Dropout(0.2, name='dropout_2'),
        k.layers.BatchNormalization(name='batch_norm_2'),
        # prediction
        k.layers.Flatten(name='flatten'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])


if __name__ == '__main__':
    # load dataset
    print('loading dataset...', end=' ', flush=True)
    data = np.load('data/data-processed-bands.npz')
    X, Y, Z = [data['x'], data['y'], data['z']]  # type: np.ndarray
    print(f'OK, X={X.shape}, Y={Y.shape}, Z={Z.shape}')

    # normalize x
    print('normalizing X...', end=' ', flush=True)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    print('OK')

    # spatial localization
    print('performing spatial localization...', end=' ', flush=True)
    X = tf.nn.max_pool(X, ksize=[1, 3, 2], strides=[1, 1, 2], padding='VALID').numpy()
    print('OK')

    print('Creating Models...', end=' ', flush=True)
    sample_shape = X.shape[1:]
    models = [create_conv_model(*sample_shape), create_lstm_model(*sample_shape)]
    print('OK')

    print('Evaluating...')
    optimizer = k.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    for model in models:
        print('=' * 100)
        print(f'Model: {model.name}')
        print(f'Total Params: {model.count_params():,}')
        print('_' * 100)
        filepath = f'weights/{model.name}.hdf5'
        save_best = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, Y, batch_size=64, epochs=200, validation_split=0.25, callbacks=[save_best])
        print('=' * 100)
    print('Done')
