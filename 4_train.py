#!/usr/bin/env python3

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras as k

import models

tf.random.set_seed(42)

if __name__ == '__main__':
    # load dataset
    print('loading dataset...', end=' ', flush=True)
    dataset = np.load('data/data-processed-bands.npz')
    X, Y, Z = [dataset['x'], dataset['y'], dataset['z']]  # type: np.ndarray
    print('OK')
    print(f'X={X.shape}, Y={Y.shape}, Z={Z.shape}')

    # normalize x
    print('normalizing X...', end=' ', flush=True)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    print('OK')

    # spatial localization
    print('performing spatial localization...', end=' ', flush=True)
    X = tf.nn.max_pool(X, ksize=[1, 3, 2], strides=[1, 1, 2], padding='VALID').numpy()
    print('OK')

    # generate time-major and channel-major data
    # NOTE: this was done to avoid repeated transposition, which is computationally expensive
    print('creating time-major and channel-major data...', end=' ', flush=True)
    X_TM = X
    X_CM = tf.transpose(X, perm=[0, 2, 3, 1, 4]).numpy()
    print('OK')

    print('Creating Models...', end=' ', flush=True)
    channel_major_models = [
        models.conv_nn_channel_major(*X_CM.shape[1:])
    ]
    time_major_models = [
        models.conv_nn_time_major(*X_TM.shape[1:]),
        models.lstm_nn(*X_TM.shape[1:])
    ]
    print('OK')

    print('Evaluating...')
    optimizer = k.optimizers.Adam(learning_rate=0.0001)
    # iterate each model type
    for models, features in [(channel_major_models, X_CM), (time_major_models, X_TM)]:
        # iterate each model in the type
        for model in models:
            filepath = f'weights/{model.name}.hdf5'
            save_best = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
            # build model
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()
            # load existing weights if exists
            if os.path.exists(filepath):
                model.load_weights(filepath)
            # fit model
            model.fit(features, Y, batch_size=64, epochs=1000, verbose=2, validation_split=0.25, callbacks=[save_best])
    print('Done')
