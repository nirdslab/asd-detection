#!/usr/bin/env python3

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == '__main__':

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras as k

    tf.random.set_seed(42)
    optimizer = k.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)


    def print_compact(line):
        if not all(x == '_' for x in line):
            print(line)


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


    # load dataset
    print('loading dataset...', end=' ', flush=True)
    data = np.load('data/data-final.npz')
    X, Y, Z = [np.abs(data['x']), data['y'], data['z']]  # type: np.ndarray
    print(f'OK, X={X.shape}, Y={Y.shape}, Z={Z.shape}')

    # normalize X
    print('normalizing data...', end=' ', flush=True)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    print('OK')

    # extract delta, theta, alpha, beta, and gamma frequency bands
    print('extracting frequency bands...', end=' ', flush=True)
    delta = X[..., 0:4]  # ( <= 4 Hz)
    theta = X[..., 3:8]  # (4 - 8 Hz)
    alpha = X[..., 7:16]  # (8 - 16 Hz)
    beta = X[..., 15:32]  # (16 - 32 Hz)
    gamma = X[..., 31:]  # ( >= 32 Hz)
    X = np.stack([np.max(x, axis=-1) for x in [delta, theta, alpha, beta, gamma]], axis=-1).astype(np.float32)  # type: np.ndarray
    print(f'OK, X={X.shape}, Y={Y.shape}, Z={Z.shape}')

    # spatial localization
    print('performing spatial localization...', end=' ', flush=True)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2], strides=[1, 1, 1], padding='VALID').numpy()
    print('OK')

    print('Creating Models...', end=' ', flush=True)
    sample_shape = X.shape[1:]
    models = [create_conv_model(*sample_shape), create_lstm_model(*sample_shape)]
    print('OK')

    print('Evaluating...')
    for model in models:
        print('=' * 100)
        print(f'Model: {model.name}')
        print(f'Total Params: {model.count_params()}')
        print('_' * 100)
        filepath = f'weights/{model.name}.hdf5'
        save_best = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, Y, batch_size=64, epochs=200, validation_split=0.25, callbacks=[save_best])
        print('=' * 100)
    print('Done')
