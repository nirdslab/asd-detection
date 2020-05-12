#!/usr/bin/env python3

import os
import sys
from typing import List

import numpy as np
import tensorflow as tf
from capsnet.losses import margin_loss
from tensorflow import keras as k

import models
from info import participants, SLICE_SHAPE_BANDS

tf.random.set_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    # parse command line arguments
    training = False
    testing = False
    if len(sys.argv) != 2:
        print('Arguments: [ train | test ]')
        exit(1)
    else:
        mode = sys.argv[1].strip().lower()
        if mode not in ['train', 'test']:
            print('Arguments: [OPTIONAL] [ train | test ]')
            exit(1)
        if mode == 'train':
            training = True
        elif mode == 'test':
            testing = True

    # load dataset
    print('loading dataset...', end=' ', flush=True)
    dataset = np.load('data/data-processed-bands.npz')

    # train-test-split on participant ID
    fraction = 0.8
    num_train = int(len(participants) * fraction)
    p_train = set(np.random.choice(participants, num_train, replace=False))

    # create test and train data
    X_TRAIN, X_TEST = np.zeros((0, *SLICE_SHAPE_BANDS)), np.zeros((0, *SLICE_SHAPE_BANDS))  # type: np.ndarray
    Y_TRAIN, Y_TEST = np.zeros(0, ), np.zeros(0, )  # type: np.ndarray
    Z_TRAIN, Z_TEST = np.zeros(0, ), np.zeros(0, )  # type: np.ndarray
    for p in participants:
        _x = dataset[f'{p}_x']
        _y = np.full(len(_x), dataset[f'{p}_y'])
        _z = np.full(len(_x), dataset[f'{p}_z'])
        if p in p_train:
            X_TRAIN = np.append(X_TRAIN, _x, axis=0)
            Y_TRAIN = np.append(Y_TRAIN, _y, axis=0)
            Z_TRAIN = np.append(Z_TRAIN, _z, axis=0)
        else:
            X_TEST = np.append(X_TEST, _x, axis=0)
            Y_TEST = np.append(Y_TEST, _y, axis=0)
            Z_TEST = np.append(Z_TEST, _z, axis=0)
    print('OK')
    print(f'TRAINING: X={X_TRAIN.shape}, Y={Y_TRAIN.shape}, Z={Z_TRAIN.shape}')
    print(f'TESTING: X={X_TEST.shape}, Y={Y_TEST.shape}, Z={Z_TEST.shape}')

    # normalize x
    print('normalizing X...', end=' ', flush=True)
    X_MAX = np.maximum(np.amax(X_TRAIN), np.amax(X_TEST))
    X_MIN = np.minimum(np.amin(X_TRAIN), np.amin(X_TEST))
    X_TRAIN = (X_TRAIN - X_MIN) / (X_MAX - X_MIN)
    X_TEST = (X_TEST - X_MIN) / (X_MAX - X_MIN)
    print('OK')

    # generate time-major and channel-major data
    # NOTE: this was done to avoid repeated transposition, which is computationally expensive
    print('creating time-major and channel-major data...', end=' ', flush=True)
    # time-major data
    DATA_TM_TRAIN = [X_TRAIN, Y_TRAIN, Z_TRAIN]  # type: List[np.ndarray]
    DATA_TM_TEST = [X_TEST, Y_TEST, Z_TEST]  # type: List[np.ndarray]
    TM_SHAPE = DATA_TM_TRAIN[0].shape[1:]
    # channel-major data
    DATA_CM_TRAIN = [tf.transpose(X_TRAIN, perm=[0, 2, 3, 1, 4]).numpy(), Y_TRAIN, Z_TRAIN]  # type: List[np.ndarray]
    DATA_CM_TEST = [tf.transpose(X_TEST, perm=[0, 2, 3, 1, 4]).numpy(), Y_TEST, Z_TEST]  # type: List[np.ndarray]
    CM_SHAPE = DATA_CM_TRAIN[0].shape[1:]
    print('OK')

    print('Creating Models...', end=' ', flush=True)

    default_loss = {'label': 'binary_crossentropy', 'score': 'mse'}
    caps_loss = {'label': margin_loss, 'score': 'mse'}

    # training models and specs (model, data, loss)
    models = [
        (models.conv_nn_tm(*TM_SHAPE), DATA_TM_TRAIN, DATA_TM_TEST, default_loss),
        (models.conv_nn_cm(*CM_SHAPE), DATA_CM_TRAIN, DATA_CM_TEST, default_loss),
        (models.capsule_nn(*TM_SHAPE), DATA_TM_TRAIN, DATA_TM_TEST, caps_loss),
        (models.lstm_nn(*TM_SHAPE), DATA_TM_TRAIN, DATA_TM_TEST, default_loss),
    ]
    print('OK')

    print('Training and Evaluation')
    optimizer = k.optimizers.Adam()
    # iterate each model type
    model = ... # type: k.Model
    for model, [x_tr, y_tr, z_tr], [x_te, y_te, z_te], loss in models:
        y_tr = k.utils.to_categorical(y_tr, num_classes=2)
        y_te = k.utils.to_categorical(y_te, num_classes=2)
        filepath = f'weights/{model.name}.hdf5'
        # build model
        model.compile(optimizer=optimizer, loss=loss, loss_weights=[1, 0.1], metrics={'label': 'accuracy', 'score': 'mse'})
        model.summary()
        # training phase
        if training:
            # load pre-trained weights when available
            if os.path.exists(filepath):
                model.load_weights(filepath)
            # train
            save_best = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)
            model.fit(x_tr, [y_tr, z_tr], batch_size=64, epochs=200, validation_data=(x_te, [y_te, z_te]), callbacks=[save_best], verbose=2)
        if testing:
            model.load_weights(filepath)
            model.evaluate(x_te, [y_te, z_te], batch_size=64)
    print('Done')
