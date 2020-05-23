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
INFO = 'Expected Arguments: [OPTIONAL] [ train | test ] [ MODEL_NAME ]'
FRACTION = 0.7

if __name__ == '__main__':
    # parse command line arguments
    assert len(sys.argv) == 3, INFO
    mode = sys.argv[1].strip().lower()
    model_name = sys.argv[2].strip().lower()
    assert mode in ['train', 'test'], INFO
    assert model_name in ['conv', 'lstm', 'caps'], INFO
    training = mode == 'train'
    testing = mode == 'test'

    # load dataset
    print('loading dataset...', end=' ', flush=True)
    dataset = np.load('data/data-processed-bands.npz')
    print('OK')

    # train-test-split on participant ID
    print('performing train-test split...', end=' ', flush=True)
    num_train = int(len(participants) * FRACTION)
    p_train = set(np.random.choice(participants, num_train, replace=False))
    print('OK')

    # create test and train data
    X_TRAIN, X_TEST = np.zeros((0, *SLICE_SHAPE_BANDS)), np.zeros((0, *SLICE_SHAPE_BANDS))  # type: np.ndarray
    Y_TRAIN, Y_TEST = np.zeros(0, ), np.zeros(0, )  # type: np.ndarray
    Z_TRAIN, Z_TEST = np.zeros(0, ), np.zeros(0, )  # type: np.ndarray
    for p in participants:
        _x = dataset[f'{p}_x']
        _y = np.full(len(_x), dataset[f'{p}_bc'])
        _z = np.full(len(_x), dataset[f'{p}_r'])
        if p in p_train:
            X_TRAIN = np.append(X_TRAIN, _x, axis=0)
            Y_TRAIN = np.append(Y_TRAIN, _y, axis=0)
            Z_TRAIN = np.append(Z_TRAIN, _z, axis=0)
        else:
            X_TEST = np.append(X_TEST, _x, axis=0)
            Y_TEST = np.append(Y_TEST, _y, axis=0)
            Z_TEST = np.append(Z_TEST, _z, axis=0)
    print(f'TRAINING: X={X_TRAIN.shape}, Y={Y_TRAIN.shape}, Z={Z_TRAIN.shape}')
    print(f'TESTING: X={X_TEST.shape}, Y={Y_TEST.shape}, Z={Z_TEST.shape}')
    D_TRAIN = [X_TRAIN, Y_TRAIN, Z_TRAIN]  # type: List[np.ndarray]
    D_TEST = [X_TEST, Y_TEST, Z_TEST]  # type: List[np.ndarray]
    print('OK')

    # training models and specs (model, data, loss)
    print('Creating Models...', end=' ', flush=True)
    conv_loss = {'l': 'categorical_crossentropy', 's': 'mae'}
    lstm_loss = {'l': 'categorical_crossentropy', 's': 'mae'}
    caps_loss = {'l': margin_loss, 's': 'mae'}
    metrics = {'l': 'acc'}
    loss_dict = {'conv': conv_loss, 'lstm': lstm_loss, 'caps': caps_loss}
    models_dict = {'conv': models.CONV, 'lstm': models.LSTM, 'caps': models.CAPS}
    print('OK')

    print('Training and Evaluation')
    model = models_dict[model_name](*SLICE_SHAPE_BANDS)
    loss = loss_dict[model_name]
    optimizer = k.optimizers.Adam(0.0005)
    [x_tr, y_tr, z_tr] = D_TRAIN
    [x_te, y_te, z_te] = D_TEST
    y_tr = k.utils.to_categorical(y_tr, num_classes=2)
    y_te = k.utils.to_categorical(y_te, num_classes=2)
    save_path = f'weights/{model.name}.hdf5'
    # build model
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1, 0.05], metrics=metrics)
    model.summary(line_length=150)
    # training phase
    if training:
        # load pre-trained weights when available
        if os.path.exists(save_path):
            model.load_weights(save_path)
        # train
        save_best = k.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)
        model.fit(x_tr, [y_tr, z_tr], batch_size=32, epochs=500, validation_data=(x_te, [y_te, z_te]), callbacks=[save_best], verbose=2)
    if testing:
        model.load_weights(save_path)
        model.evaluate(x_te, [y_te, z_te], batch_size=32, verbose=2)
    print('Done')
