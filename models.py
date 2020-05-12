#!/usr/bin/env python3

from capsnet.layers import ConvCaps2D, DenseCaps
from capsnet.nn import squash, norm, mask_cid
from tensorflow import keras as k
from tensorflow.keras import layers as kl, models as km


def lstm_nn(timesteps, ch_rows, ch_cols, bands):
    """
    Generate LSTM NN model, with temporal dimension addressed first
    """
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))

    # == intermediate layer(s) ==
    ml = kl.TimeDistributed(kl.Flatten(), name='eeg')(il)
    # lstm block 1
    ml = kl.LSTM(32, return_sequences=True, kernel_regularizer='l1_l2', dropout=0.2, name='lstm_1')(ml)
    # lstm block 2
    ml = kl.LSTM(64, kernel_regularizer='l1_l2', dropout=0.2, name='lstm_2')(ml)

    # == output layer(s) ==
    ol_c = kl.Dense(2, activation='sigmoid', kernel_regularizer='l1_l2', name='label')(ml)
    ol_r = kl.Dense(1, activation='relu', kernel_regularizer='l1_l2', name='score')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[ol_c, ol_r], name='asd_lstm')


def conv_nn_tm(timesteps, ch_rows, ch_cols, bands):
    """
    Generate 1D-convolution NN model, with temporal dimension addressed first
    """
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))

    # == intermediate layer(s) ==
    ml = kl.TimeDistributed(kl.Flatten(), name='eeg')(il)
    # convolution 1
    ml = kl.Conv1D(filters=64, kernel_size=10, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_1')(ml)
    ml = kl.MaxPooling1D(name='pool_1')(ml)
    ml = kl.Dropout(0.2, name='dropout_1')(ml)
    ml = kl.BatchNormalization(name='b_norm_1')(ml)
    # convolution 2
    ml = kl.Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_2')(ml)
    ml = kl.MaxPooling1D(name='pool_2')(ml)
    ml = kl.Dropout(0.2, name='dropout_2')(ml)
    ml = kl.BatchNormalization(name='b_norm_2')(ml)
    # flatten
    ml = kl.Flatten(name='flatten')(ml)

    # == output layer(s) ==
    ol_c = kl.Dense(2, activation='sigmoid', kernel_regularizer='l1_l2', name='label')(ml)
    ol_r = kl.Dense(1, activation='relu', kernel_regularizer='l1_l2', name='score')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[ol_c, ol_r], name='asd_conv_tm')


def conv_nn_cm(ch_rows, ch_cols, timesteps, bands):
    """
    Generate 1D-convolution NN model, with channel dimension addressed first
    """
    # == input layer(s) ==
    il = kl.Input(shape=(ch_rows, ch_cols, timesteps, bands))

    # == intermediate layer(s) ==
    ml = kl.Reshape(target_shape=(ch_rows * ch_cols, timesteps, bands), name='eeg')(il)
    # convolution 1
    ml = kl.TimeDistributed(kl.Conv1D(filters=64, kernel_size=10, activation='relu', kernel_regularizer='l1_l2', padding='same'), name='conv_1')(ml)
    ml = kl.TimeDistributed(kl.MaxPooling1D(), name='pool_1')(ml)
    ml = kl.TimeDistributed(kl.Dropout(0.2), name='dropout_1')(ml)
    ml = kl.TimeDistributed(kl.BatchNormalization(), name='b_norm_1')(ml)
    # convolution 2
    ml = kl.TimeDistributed(kl.Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer='l1_l2', padding='same'), name='conv_2')(ml)
    ml = kl.TimeDistributed(kl.MaxPooling1D(), name='pool_2')(ml)
    ml = kl.TimeDistributed(kl.Dropout(0.2), name='dropout_2')(ml)
    ml = kl.TimeDistributed(kl.BatchNormalization(), name='b_norm_2')(ml)
    # flatten
    ml = k.layers.Flatten(name='flatten')(ml)

    # == output layer(s) ==
    ol_c = kl.Dense(2, activation='sigmoid', kernel_regularizer='l1_l2', name='label')(ml)
    ol_r = kl.Dense(1, activation='relu', kernel_regularizer='l1_l2', name='score')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[ol_c, ol_r], name='asd_conv_cm')


def capsule_nn(timesteps, ch_rows, ch_cols, bands):
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))

    # == intermediate layer(s) ==
    ml = kl.Reshape(target_shape=(timesteps, ch_rows * ch_cols, bands), name='eeg')(il)
    # initial convolution
    ml = kl.Conv2D(filters=32, kernel_size=(10, 1), strides=(2, 1), activation='relu', name='conv')(ml)
    # convert to capsule domain
    ml = ConvCaps2D(filters=8, filter_dims=4, kernel_size=(5, 1), strides=(2, 1), name='conv_caps')(ml)
    ml = kl.Lambda(squash)(ml)
    # dense capsule layer with dynamic routing
    ml = DenseCaps(caps=2, caps_dims=8, routing_iter=3, name='dense_caps')(ml)
    ml = kl.Lambda(squash)(ml)
    # select capsule with highest activity
    cl = kl.Lambda(mask_cid)(ml)

    # == output layer(s) ==
    label = kl.Lambda(norm, name='label')(ml)
    score = kl.Dense(1, name='score')(cl)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[label, score], name='asd_caps_nn')
