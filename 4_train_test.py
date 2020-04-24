#!/usr/bin/env python3

import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras as k

import models

tf.random.set_seed(42)


def train_dev_test_split(_x: np.ndarray, _y: np.ndarray, _z: np.ndarray, r_dev: float, r_test: float, rand_state: int):
    """
    train/dev/test split at given ratios
    """
    from sklearn.model_selection import train_test_split
    assert r_dev + r_test < 1
    r1 = r_dev + r_test
    r2 = r_test / r1
    _x_tr, _x_ts, _y_tr, _y_ts, _z_tr, _z_ts = train_test_split(_x, _y, _z, test_size=r1, shuffle=True, random_state=rand_state)
    _x_dv, _x_ts, _y_dv, _y_ts, _z_dv, _z_ts = train_test_split(_x_ts, _y_ts, _z_ts, test_size=r2, shuffle=True, random_state=rand_state)
    result = [_x_tr, _x_dv, _x_ts], [_y_tr, _y_dv, _y_ts], [_z_tr, _z_dv, _z_ts]  # type: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
    return result


if __name__ == '__main__':
    # parse command line arguments
    training = True
    testing = True
    if len(sys.argv) > 2:
        print('Arguments: [ train | test ]')
        exit(1)
    elif len(sys.argv) == 2:
        mode = sys.argv[1].strip().lower()
        if mode not in ['train', 'test']:
            print('Arguments: [OPTIONAL] [ train | test ]')
            exit(1)
        if mode == 'train':
            testing = False
        elif mode == 'test':
            training = False

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
    # time-major data
    DATA_TM = [X, Y, Z]
    TM_SAMPLE_SHAPE = DATA_TM[0].shape[1:]
    # channel-major data
    DATA_CM = [tf.transpose(X, perm=[0, 2, 3, 1, 4]).numpy(), Y, Z]  # type: List[np.ndarray]
    CM_SAMPLE_SHAPE = DATA_CM[0].shape[1:]
    print('OK')

    print('Creating Models...', end=' ', flush=True)
    time_major_models = [
        models.conv_nn_time_major(*TM_SAMPLE_SHAPE),
        models.lstm_nn(*TM_SAMPLE_SHAPE)
    ]
    channel_major_models = [
        models.conv_nn_channel_major(*CM_SAMPLE_SHAPE)
    ]
    print('OK')

    print('Training and Evaluation')
    optimizer = k.optimizers.Adam(learning_rate=0.0001)
    # iterate each model type
    for models, [x, y, z] in [(channel_major_models, DATA_CM), (time_major_models, DATA_TM)]:
        # train-dev-test split at 60-20-20 ratio
        [x_tr, x_dv, x_ts], [y_tr, y_dv, y_ts], [z_tr, z_dv, z_ts] = train_dev_test_split(x, y, z, r_dev=0.2, r_test=0.2, rand_state=42)
        for model in models:
            filepath = f'weights/{model.name}.hdf5'
            # build model
            model.compile(optimizer=optimizer,
                          loss={'label': 'binary_crossentropy', 'score': 'mse'},
                          loss_weights={'label': 1.0, 'score': 2.0},
                          metrics={'label': 'accuracy', 'score': 'mse'})
            model.summary()
            # training phase
            if training:
                # load pre-trained weights when available
                if os.path.exists(filepath):
                    model.load_weights(filepath)
                # train
                save_best = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
                model.fit(x_tr, [y_tr, z_tr], batch_size=64, epochs=1000, validation_data=(x_dv, [y_dv, z_dv]), callbacks=[save_best])
            # testing phase
            if testing:
                model.load_weights(filepath)
                model.evaluate(x_ts, [y_ts, z_ts])
    print('Done')
